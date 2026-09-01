import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from server import crud, schemas
from server.config import AVATARS_DIR
from server.utils.auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    get_current_user,
    get_optional_current_user,
    get_password_hash,
    verify_google_id_token,
    verify_password,
    verify_refresh_token,
)
from server.utils.common import validate_email_send_otp
from server.services.audit_logger import AuditLogger
from server.services.pricing_service import PricingService
from server.services.settings_service import SettingsService
from server.services.credit_service import CreditService

from ..deps import get_db

TOKEN_EXPIRE_DELTA = 20160

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

user_router = APIRouter(prefix="/user", tags=["User"])


def build_login_response(
    user, *, message: str = "You have successfully logged in."
) -> JSONResponse:
    """Issue access + refresh JWTs for ``user`` and build the standard login body.

    Shared by every endpoint that logs a user in (email/password and Google) so
    token generation and the response shape stay identical.
    """
    access_token_expires = timedelta(minutes=TOKEN_EXPIRE_DELTA)
    access_token = create_access_token(
        data={"sub": user.email, "id": user.id}, expires_delta=access_token_expires
    )

    # create refresh token with longer expiry
    refresh_expires = timedelta(minutes=TOKEN_EXPIRE_DELTA * 2)
    refresh_token = create_refresh_token(
        data={"sub": user.email, "id": user.id}, expires_delta=refresh_expires
    )

    disp_name = user.display_name or getattr(user, 'username', None) or user.email.split('@')[0]
    response = schemas.RespUser(
        id=user.id,
        email=user.email,
        avatar=user.avatar,
        display_name=disp_name,
        is_verified=user.is_verified,
        providers=user.providers,
        role=user.role,
        created_at=str(user.created_at),
        updated_at=str(user.updated_at),
        token=schemas.Token(
            access_token=access_token, refresh_token=refresh_token, token_type="bearer"
        ),
    )

    AuditLogger.log(
        action="USER_LOGIN",
        user_id=str(user.id),
        status="200",
        metadata={"email": user.email, "providers": user.providers},
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": response.model_dump(mode="json"),
            "message": message,
        },
    )


@user_router.post("/login")
async def login_for_access_token(
    login_details: schemas.UserCreateInput, db: Session = Depends(get_db)
):
    logger.info("login attempt: email=%s", login_details.email)
    user, error_msg = authenticate_user(db, login_details.email, login_details.password)
    if not user:
        logger.warning(
            "login failed: email=%s - incorrect credentials", login_details.email
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_msg,
            headers={"WWW-Authenticate": "Bearer"},
        )
    logger.info("login success: user_id=%s email=%s", user.id, user.email)
    return build_login_response(user)


@user_router.post("/google-login")
async def google_login(data: schemas.IdTokenLoginInput, db: Session = Depends(get_db)):
    """Authenticate with a Google ID token (Google OAuth 2.0).

    Verifies the token with Google, then logs the user in — creating the account
    on first sign-in. Reuses the same JWT tokens and response shape as
    email/password login.
    """

    idinfo = verify_google_id_token(data.id_token)

    email = (idinfo.get("email") or "").lower().strip()
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google account did not provide an email address.",
        )
    if not idinfo.get("email_verified", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google email address is not verified.",
        )

    google_id = idinfo.get("sub")
    google_name = idinfo.get("name") or idinfo.get("given_name")
    fallback_name = google_name or (email.split("@")[0] if email else "")
    picture = idinfo.get("picture")

    # Match an existing account by email (unique) — this also links Google to a
    # pre-existing email/password account. Fall back to the Google subject id.
    user = crud.user.get_by_email(db, email=email)
    if not user and google_id:
        user = crud.user.get_by_google_id(db, google_id=google_id)

    if user:
        # Backfill any missing Google fields on the existing account, then log in.
        updates = {}
        if google_id and not user.google_id:
            updates["google_id"] = google_id
        if not user.display_name:
            updates["display_name"] = fallback_name or getattr(user, 'username', None) or email.split("@")[0]
        if picture and not user.avatar:
            updates["avatar"] = picture
        if not user.is_verified:
            updates["is_verified"] = True
        if updates:
            user = crud.user.update(
                db, db_obj=user, obj_in=schemas.UserUpdate(**updates)
            )
        logger.info("google login: existing user_id=%s email=%s", user.id, email)
    else:
        user = crud.user.create(
            db,
            obj_in=schemas.UserBase(
                email=email,
                google_id=google_id,
                display_name=fallback_name,
                avatar=picture or "",
                is_verified=True,
                providers="google",
                hashed_password=None,
            ),
        )
        logger.info("google login: created user_id=%s email=%s", user.id, email)

    return build_login_response(
        user, message="You have successfully logged in with Google."
    )


@user_router.get("/profile", response_model=schemas.UserProfile)
@user_router.post("/profile", response_model=schemas.UserProfile)
async def get_profile(current_user=Depends(get_current_user)):
    disp_name = current_user.display_name or getattr(current_user, 'username', None) or current_user.email.split('@')[0]
    uname = getattr(current_user, 'username', None) or current_user.email.split('@')[0]
    response = schemas.UserProfile(
        id=current_user.id,
        email=current_user.email,
        avatar=current_user.avatar,
        role=current_user.role,
        display_name=disp_name,
        username=uname,
    )
    logger.info("get_profile: user_id=%s", getattr(current_user, "id", None))
    AuditLogger.log(
        action="USER_PROFILE",
        user_id=str(current_user.id),
        endpoint="/v1/user/profile",
        http_method="POST",
        status="200",
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": response.model_dump(mode="json"),
            "message": "You have successfully logged in.",
        },
    )


@user_router.post("/signup")
async def sign_up(data: schemas.UserCreateInput, db: Session = Depends(get_db)):
    email = data.email.lower().strip()
    logger.info("signup attempt: email=%s", email)
    user = crud.user.get_by_email(db, email=email)

    if user:
        if user.is_verified:
            logger.warning("signup aborted: email=%s already verified", email)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You already have an account with us. please continue with login.",
            )

    else:
        hashed_password = get_password_hash(data.password)
        disp_name = data.display_name
        user = crud.user.create(
            db,
            obj_in=schemas.UserBase(
                email=data.email,
                hashed_password=hashed_password,
                display_name=disp_name,
            ),
        )
        logger.info("signup created: user_id=%s email=%s", user.id, email)

    otp_sent = await validate_email_send_otp(db, email, user.id)
    if otp_sent:
        AuditLogger.log(
            action="USER_SIGNUP",
            user_id=str(user.id),
            endpoint="/v1/user/signup",
            http_method="POST",
            status="200",
            metadata={"email": email},
        )
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "error": None,
                "data": None,
                "message": "OTP sent to your email, please verify your email to continue.",
            },
        )


@user_router.post("/verify-otp")
async def verify_otp(data: schemas.VerifyOTP, db: Session = Depends(get_db)):
    email = data.email.lower().strip()
    otp_sent = data.otp.lower().strip()

    logger.info("verify_otp attempt: email=%s", email)

    email_otp_obj = crud.email_otp.get_by_email(db, email=email)
    user_obj = crud.user.get_by_email(db, email=email)

    if user_obj:
        if user_obj.is_verified:
            logger.warning("verify_otp: email=%s already verified", email)
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "success": True,
                    "message": "Your email is already verified.",
                    "data": None,
                },
            )

    if not email_otp_obj:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="First proceed via sending OTP request.",
        )

    otp = email_otp_obj.otp
    time_now = datetime.utcnow()

    if (((time_now - email_otp_obj.updated_at).total_seconds()) / 60) < 10:
        if str(otp_sent) == str(otp):
            user_obj = crud.user.update(
                db, db_obj=user_obj, obj_in=schemas.UserUpdate(is_verified=True)
            )
            # crud.email_otp.remove(db, id=email_otp_obj.id)
            access_token_expires = timedelta(minutes=TOKEN_EXPIRE_DELTA)
            access_token = create_access_token(
                data={"sub": user_obj.email, "id": user_obj.id},
                expires_delta=access_token_expires,
            )
            # create refresh token with longer expiry
            refresh_expires = timedelta(minutes=TOKEN_EXPIRE_DELTA * 2)
            refresh_token = create_refresh_token(
                data={"sub": user_obj.email, "id": user_obj.id},
                expires_delta=refresh_expires,
            )
            logger.info("verify_otp success: email=%s user_id=%s", email, user_obj.id)
            AuditLogger.log(
                action="USER_VERIFY_OTP",
                user_id=str(user_obj.id),
                endpoint="/v1/user/verify-otp",
                http_method="POST",
                status="200",
                metadata={"email": email},
            )
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "error": None,
                    "data": {
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                        "token_type": "bearer",
                    },
                    "message": "OTP verified successfully, thanks for signup.",
                },
            )
        else:
            logger.warning("verify_otp failed: email=%s otp_mismatch", email)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OTP mismatched, Please provide correct OTP.",
            )
    else:
        crud.email_otp.remove(db, id=email_otp_obj.id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP is expired, please proceed new OTP.",
        )


@user_router.post("/signup-resend-otp")
async def resend_otp(data: schemas.EmailSchema, db: Session = Depends(get_db)):
    email = data.email.lower().strip()
    logger.info("resend_otp requested: email=%s", email)
    user_obj = crud.user.get_by_email(db, email=email)
    if not user_obj:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="First proceed via sending OTP request.",
        )

    if user_obj and user_obj.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Your email is already verified.",
        )

    otp_sent = await validate_email_send_otp(db, email, user_obj.id)
    if otp_sent:
        logger.info("resend_otp sent: email=%s user_id=%s", email, user_obj.id)
        AuditLogger.log(
            action="USER_RESEND_OTP",
            user_id=str(user_obj.id),
            endpoint="/v1/user/signup-resend-otp",
            http_method="POST",
            status="200",
            metadata={"email": email},
        )
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "error": None,
                "data": None,
                "message": "OTP sent to your email, please verify your email to continue.",
            },
        )


@user_router.post("/change-password")
async def change_password(
    data: schemas.ChangePassword,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    logger.info(
        "change_password attempt: user_id=%s", getattr(current_user, "id", None)
    )
    old_password = data.old_password
    new_password = data.new_password
    user_obj = crud.user.get_by_email(db, email=current_user.email)
    if not user_obj:
        logger.warning(
            "change_password failed: user not found user_id=%s",
            getattr(current_user, "id", None),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="User not found."
        )

    if old_password == new_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Old password and new password are same.",
        )

    if not verify_password(old_password, user_obj.hashed_password):
        logger.warning(
            "change_password failed: incorrect old password user_id=%s", user_obj.id
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Old password is incorrect."
        )

    hashed_password = get_password_hash(new_password)
    user_obj = crud.user.update(
        db,
        db_obj=user_obj,
        obj_in=schemas.UserUpdate(hashed_password=hashed_password),
    )
    logger.info("change_password success: user_id=%s", user_obj.id)
    access_token_expires = timedelta(minutes=TOKEN_EXPIRE_DELTA)
    access_token = create_access_token(
        data={"sub": user_obj.email, "id": user_obj.id},
        expires_delta=access_token_expires,
    )

    AuditLogger.log(
        action="USER_CHANGE_PASSWORD",
        user_id=str(user_obj.id),
        endpoint="/v1/user/change-password",
        http_method="POST",
        status="200",
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": {"token": access_token, "token_type": "bearer"},
            "message": "Password changed successfully.",
        },
    )


@user_router.post("/forgot-password-send-otp")
async def forgot_password_send_otp(
    data: schemas.EmailSchema, db: Session = Depends(get_db)
):
    email = data.email.lower().strip()
    logger.info("forgot_password_send_otp requested: email=%s", email)
    user_obj = crud.user.get_by_email(db, email=email)
    if not user_obj:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="User not found."
        )

    otp_sent = await validate_email_send_otp(db, email, user_obj.id)
    if otp_sent:
        logger.info(
            "forgot_password_send_otp sent: email=%s user_id=%s", email, user_obj.id
        )
        AuditLogger.log(
            action="USER_FORGOT_PASSWORD_SEND_OTP",
            user_id=str(user_obj.id),
            endpoint="/v1/user/forgot-password-send-otp",
            http_method="POST",
            status="200",
            metadata={"email": email},
        )
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "error": None,
                "data": None,
                "message": "OTP sent to your email, please verify your email to continue.",
            },
        )


@user_router.post("/reset-forgot-password")
async def reset_forgot_password(
    data: schemas.ResetPassword, db: Session = Depends(get_db)
):
    email = data.email.lower().strip()
    otp_sent = data.otp.lower().strip()
    new_password = data.new_password

    logger.info("reset_forgot_password attempt: email=%s", email)

    email_otp_obj = crud.email_otp.get_by_email(db, email=email)
    user_obj = crud.user.get_by_email(db, email=email)
    if not user_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This email is not associated with any account, please sign up.",
        )
    if not email_otp_obj:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="First proceed via sending OTP request.",
        )

    otp = email_otp_obj.otp
    time_now = datetime.utcnow()

    if (((time_now - email_otp_obj.updated_at).total_seconds()) / 60) < 10:
        if str(otp_sent) == str(otp):
            hashed_password = get_password_hash(new_password)
            if user_obj.hashed_password and verify_password(new_password, user_obj.hashed_password):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="New password is same as your old password.",
                )

            user_obj = crud.user.update(
                db,
                db_obj=user_obj,
                obj_in=schemas.UserUpdate(hashed_password=hashed_password),
            )

            crud.email_otp.remove(db, id=email_otp_obj.id)
            logger.info(
                "reset_forgot_password success: email=%s user_id=%s", email, user_obj.id
            )
            AuditLogger.log(
                action="USER_RESET_FORGOT_PASSWORD",
                user_id=str(user_obj.id),
                endpoint="/v1/user/reset-forgot-password",
                http_method="POST",
                status="200",
                metadata={"email": email},
            )
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "error": None,
                    "data": None,
                    "message": "OTP matched, please proceed further.",
                },
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OTP mismatched, Please provide correct OTP.",
            )
    else:
        logger.warning("reset_forgot_password failed: otp_expired email=%s", email)
        crud.email_otp.remove(db, id=email_otp_obj.id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP is expired, please proceed with new OTP.",
        )


@user_router.post("/update-profile")
async def update_profile(
    data: schemas.UserUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    logger.info("update_profile attempt: user_id=%s", getattr(current_user, "id", None))
    user_obj = crud.user.get_by_email(db, email=current_user.email)
    if not user_obj:
        logger.warning(
            "update_profile failed: user not found user_id=%s",
            getattr(current_user, "id", None),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="User not found."
        )

    # Avoid logging sensitive fields like hashed_password
    changed = {
        k: v
        for k, v in data.model_dump(exclude_unset=True).items()
        if k != "hashed_password"
    }
    logger.debug("update_profile: user_id=%s changes=%s", user_obj.id, changed)

    if getattr(data, "name", None) and not data.display_name:
        data.display_name = data.name

    user_obj = crud.user.update(db, db_obj=user_obj, obj_in=data)
    disp_name = user_obj.display_name or getattr(user_obj, 'username', None) or user_obj.email.split('@')[0]
    uname = getattr(user_obj, 'username', None) or user_obj.email.split('@')[0]
    response = schemas.UserProfile(
        id=user_obj.id,
        email=user_obj.email,
        avatar=user_obj.avatar,
        role=user_obj.role,
        display_name=disp_name,
        username=uname,
    )
    logger.info("update_profile success: user_id=%s", user_obj.id)
    AuditLogger.log(
        action="USER_UPDATE_PROFILE",
        user_id=str(user_obj.id),
        endpoint="/v1/user/update-profile",
        http_method="POST",
        status="200",
    )
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": response.model_dump(mode="json"),
            "message": "Profile updated successfully.",
        },
    )


ALLOWED_AVATAR_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".svg",
    ".bmp",
    ".ico",
}
ALLOWED_AVATAR_MIMES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/svg+xml",
    "image/bmp",
    "image/x-icon",
    "image/vnd.microsoft.icon",
}


@user_router.post("/upload-avatar")
@user_router.post("/avatar")
async def upload_avatar(
    request: Request,
    file: Optional[UploadFile] = File(None),
    avatar: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Upload and persist user avatar image via multipart/form-data."""
    uploaded_file = file or avatar or image
    if not uploaded_file:
        form_data = await request.form()
        for field in ("file", "avatar", "image", "doc"):
            val = form_data.get(field)
            if isinstance(val, UploadFile):
                uploaded_file = val
                break
        if not uploaded_file:
            for val in form_data.values():
                if isinstance(val, UploadFile):
                    uploaded_file = val
                    break

    if not uploaded_file or not getattr(uploaded_file, "filename", None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No avatar image file provided.",
        )

    ext = Path(uploaded_file.filename).suffix.lower()
    content_type = (uploaded_file.content_type or "").lower()

    is_valid_type = (
        content_type in ALLOWED_AVATAR_MIMES
        or content_type.startswith("image/")
        or ext in ALLOWED_AVATAR_EXTENSIONS
    )
    if not is_valid_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image (JPG, PNG, WebP, GIF, SVG).",
        )

    contents = await uploaded_file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty.",
        )

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Avatar image size exceeds the 10MB limit.",
        )

    AVATARS_DIR.mkdir(parents=True, exist_ok=True)

    if ext not in ALLOWED_AVATAR_EXTENSIONS:
        ext = ".png"

    safe_filename = f"avatar_{current_user.id}_{uuid.uuid4().hex[:8]}{ext}"
    target_path = AVATARS_DIR / safe_filename
    target_path.write_bytes(contents)

    avatar_url = f"/static/avatars/{safe_filename}"

    user_obj = crud.user.get_by_id(db, id=str(current_user.id))
    if not user_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    user_obj.avatar = avatar_url
    crud.user.save(db, db_obj=user_obj)

    disp_name = (
        user_obj.display_name
        or getattr(user_obj, "username", None)
        or user_obj.email.split("@")[0]
    )
    uname = getattr(user_obj, "username", None) or user_obj.email.split("@")[0]
    response = schemas.UserProfile(
        id=user_obj.id,
        email=user_obj.email,
        avatar=user_obj.avatar,
        role=user_obj.role,
        display_name=disp_name,
        username=uname,
        subscription_tier=user_obj.subscription_tier or "free",
        daily_credits=user_obj.daily_credits or 100,
        credits_remaining=user_obj.credits_remaining or 100,
        subscription_status=user_obj.subscription_status or "active",
        subscription_period=user_obj.subscription_period or "monthly",
    )

    logger.info("upload_avatar success: user_id=%s avatar=%s", user_obj.id, avatar_url)
    AuditLogger.log(
        action="USER_UPLOAD_AVATAR",
        user_id=str(user_obj.id),
        endpoint="/v1/user/upload-avatar",
        http_method="POST",
        status="200",
        metadata={"filename": safe_filename, "avatar_url": avatar_url},
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": response.model_dump(mode="json"),
            "message": "Avatar uploaded successfully.",
        },
    )


@user_router.delete("/avatar")
@user_router.post("/delete-avatar")
async def delete_avatar(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Remove user's custom avatar."""
    user_obj = crud.user.get_by_id(db, id=str(current_user.id))
    if not user_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    user_obj.avatar = ""
    crud.user.save(db, db_obj=user_obj)

    disp_name = (
        user_obj.display_name
        or getattr(user_obj, "username", None)
        or user_obj.email.split("@")[0]
    )
    uname = getattr(user_obj, "username", None) or user_obj.email.split("@")[0]
    response = schemas.UserProfile(
        id=user_obj.id,
        email=user_obj.email,
        avatar=user_obj.avatar,
        role=user_obj.role,
        display_name=disp_name,
        username=uname,
        subscription_tier=user_obj.subscription_tier or "free",
        daily_credits=user_obj.daily_credits or 100,
        credits_remaining=user_obj.credits_remaining or 100,
        subscription_status=user_obj.subscription_status or "active",
        subscription_period=user_obj.subscription_period or "monthly",
    )

    AuditLogger.log(
        action="USER_DELETE_AVATAR",
        user_id=str(user_obj.id),
        endpoint="/v1/user/avatar",
        http_method="DELETE",
        status="200",
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": response.model_dump(mode="json"),
            "message": "Avatar removed successfully.",
        },
    )


@user_router.post("/refresh-token")
async def refresh_access_token(
    data: schemas.RefreshTokenSchema, db: Session = Depends(get_db)
):
    rt = data.refresh_token
    payload = verify_refresh_token(rt)
    if not payload:
        logger.warning("refresh_access_token failed: invalid_refresh_token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = crud.user.get_by_id(db, id=payload.get("id"))
    if not user:
        logger.warning(
            "refresh_access_token failed: user_not_found for token id=%s",
            payload.get("id"),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found for token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info("refresh_access_token: user_id=%s", user.id)

    access_token_expires = timedelta(minutes=TOKEN_EXPIRE_DELTA)
    access_token = create_access_token(
        data={"sub": user.email, "id": user.id}, expires_delta=access_token_expires
    )

    AuditLogger.log(
        action="USER_REFRESH_TOKEN",
        user_id=str(user.id),
        endpoint="/v1/user/refresh-token",
        http_method="POST",
        status="200",
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": {
                "access_token": access_token,
                "token_type": "bearer",
            },
            "message": "Refresh token generated successfully.",
        },
    )

@user_router.get("/plans")
async def get_subscription_plans(
    current_user=Depends(get_optional_current_user),
):
    """Retrieve all available subscription tiers, pricing, features, and limits.

    Supports optional Bearer token authentication to identify the current user's
    active plan and indicate which plan they are currently subscribed to.
    """
    user_tier = getattr(current_user, "subscription_tier", None) if current_user else None
    pricing_data = PricingService.get_pricing_overview(current_user_tier=user_tier)
    
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": pricing_data.model_dump(mode="json"),
            "message": "Subscription plans and pricing retrieved successfully.",
        },
    )


@user_router.get("/subscription")
async def get_user_subscription(
    current_user=Depends(get_current_user),
):
    """Get the active subscription, credits, and renewal details for the logged-in user."""
    logger.info("get_user_subscription: user_id=%s", current_user.id)
    sub_data = PricingService.get_user_subscription(current_user)
    
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": sub_data.model_dump(mode="json"),
            "message": "User subscription details retrieved successfully.",
        },
    )


@user_router.post("/subscription/upgrade")
async def upgrade_user_subscription(
    data: schemas.UpgradePlanRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Upgrade or switch the logged-in user's subscription tier.

    Validates target tier ID, updates user's subscription status, period,
    and refreshes daily credit allowance.
    """
    logger.info(
        "upgrade_user_subscription attempt: user_id=%s target_plan=%s cycle=%s",
        current_user.id,
        data.plan_id,
        data.billing_cycle,
    )
    try:
        updated_sub = PricingService.upgrade_user_plan(
            db=db,
            user=current_user,
            plan_id=data.plan_id,
            billing_cycle=data.billing_cycle or "monthly",
        )
    except ValueError as e:
        logger.warning(
            "upgrade_user_subscription invalid parameter: user_id=%s error=%s",
            current_user.id,
            str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    AuditLogger.log(
        action="USER_UPGRADE_SUBSCRIPTION",
        user_id=str(current_user.id),
        endpoint="/v1/user/subscription/upgrade",
        http_method="POST",
        status="200",
        metadata={
            "plan_id": data.plan_id,
            "billing_cycle": data.billing_cycle,
            "new_tier": updated_sub.subscription_tier,
            "daily_credits": updated_sub.daily_credits,
        },
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": updated_sub.model_dump(mode="json"),
            "message": f"Successfully upgraded subscription to {data.plan_id.capitalize()}.",
        },
    )


@user_router.get("/credits")
async def get_user_credits(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get the verified daily credit allowance (100 for free tier), usage today, remaining balance, and reset timer."""
    user_obj = crud.user.get_by_id(db, id=str(current_user.id)) or current_user
    credits_data = CreditService.get_user_credits(db=db, user=user_obj)
    
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": credits_data.model_dump(mode="json"),
            "message": "User credits balance retrieved successfully.",
        },
    )


@user_router.post("/credits/consume")
async def consume_user_credits(
    data: schemas.CreditConsumeRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Deduct credits from user daily balance for AI operations, preventing negative balance."""
    user_obj = crud.user.get_by_id(db, id=str(current_user.id))
    if not user_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    try:
        updated_credits = CreditService.consume_credits(
            db=db,
            user=user_obj,
            amount=data.amount,
            purpose=data.purpose or "ai_generation",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": updated_credits.model_dump(mode="json"),
            "message": f"Successfully consumed {data.amount} credit(s).",
        },
    )


@user_router.post("/credits/reset")
async def reset_user_credits(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Reset user daily credits back to full quota."""
    user_obj = crud.user.get_by_id(db, id=str(current_user.id))
    if not user_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    updated_credits = CreditService.reset_user_credits(db=db, user=user_obj, force=True)
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": updated_credits.model_dump(mode="json"),
            "message": "User credits reset to daily limit successfully.",
        },
    )




@user_router.get("/credits/history")
async def get_user_credits_history(
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Retrieve transaction history of credit usage and daily resets for the user."""
    history = CreditService.get_credit_history(db=db, user_id=str(current_user.id), limit=limit)
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": [h.model_dump(mode="json") for h in history],
            "message": "Credit history retrieved successfully.",
        },
    )



@user_router.get("/settings")
async def get_user_settings(
    current_user=Depends(get_current_user),
):
    """Retrieve full profile settings, preferences, connectors, privacy, and credits for the logged-in user."""
    logger.info("get_user_settings: user_id=%s", current_user.id)
    settings_data = SettingsService.get_user_settings(current_user)
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": settings_data.model_dump(mode="json"),
            "message": "User settings retrieved successfully.",
        },
    )


@user_router.post("/settings")
async def update_user_settings(
    data: schemas.UserSettingsUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Update profile settings, personalization traits, language, and privacy preferences."""
    logger.info("update_user_settings attempt: user_id=%s", current_user.id)
    user_obj = crud.user.get_by_id(db, id=str(current_user.id))
    if not user_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    updated_settings = SettingsService.update_user_settings(
        db=db,
        user=user_obj,
        update_data=data,
    )

    AuditLogger.log(
        action="USER_UPDATE_SETTINGS",
        user_id=str(user_obj.id),
        endpoint="/v1/user/settings",
        http_method="POST",
        status="200",
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": updated_settings.model_dump(mode="json"),
            "message": "User settings updated successfully.",
        },
    )


@user_router.post("/help")
async def submit_help_query(
    data: schemas.HelpQueryCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_current_user),
):
    """Submit a support query or complaint form.

    Dispatches notification to company email (hello@ojasyalabs.com) and sends
    acknowledgment confirmation to the user.
    """
    try:
        ticket_resp = await SettingsService.submit_help_query(
            db=db,
            query_in=data,
            current_user=current_user,
        )
    except ValueError as e:
        logger.warning("submit_help_query validation error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": ticket_resp.model_dump(mode="json"),
            "message": "Your query has been submitted successfully. Our team will get back to you shortly.",
        },
    )


@user_router.get("/help")
async def get_user_help_queries(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Retrieve all submitted support queries for the current user."""

    queries = crud.help_query.get_by_user(db, user_id=str(current_user.id))
    serialized = [schemas.HelpQueryResponse.model_validate(q).model_dump(mode="json") for q in queries]

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": serialized,
            "message": "Help queries retrieved successfully.",
        },
    )


@user_router.get("/about")
async def get_about_us():
    """Retrieve company, website (https://www.ojasyalabs.com), and platform information."""
    about_info = SettingsService.get_about_us()
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": about_info.model_dump(mode="json"),
            "message": "About us information retrieved successfully.",
        },
    )


