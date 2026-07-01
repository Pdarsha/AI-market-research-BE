import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from server import crud, schemas
from server.utils.auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    get_current_user,
    get_password_hash,
    verify_password,
    verify_refresh_token,
)
from server.utils.common import validate_email_send_otp

from ..deps import get_db

TOKEN_EXPIRE_DELTA = 20160

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

user_router = APIRouter(prefix="/user", tags=["User"])


@user_router.post("/login")
async def login_for_access_token(
    login_details: schemas.UserCreateInput, db: Session = Depends(get_db)
):
    logger.info("login attempt: email=%s", login_details.email)
    user = authenticate_user(db, login_details.email, login_details.password)
    if not user:
        logger.warning(
            "login failed: email=%s - incorrect credentials", login_details.email
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=TOKEN_EXPIRE_DELTA)
    access_token = create_access_token(
        data={"sub": user.email, "id": user.id}, expires_delta=access_token_expires
    )

    # create refresh token with longer expiry
    refresh_expires = timedelta(minutes=TOKEN_EXPIRE_DELTA * 2)
    refresh_token = create_refresh_token(
        data={"sub": user.email, "id": user.id}, expires_delta=refresh_expires
    )

    response = schemas.RespUser(
        id=user.id,
        email=user.email,
        avatar=user.avatar,
        is_verified=user.is_verified,
        providers=user.providers,
        role=user.role,
        created_at=str(user.created_at),
        updated_at=str(user.updated_at),
        token=schemas.Token(
            access_token=access_token, refresh_token=refresh_token, token_type="bearer"
        ),
    )
    logger.info("login success: user_id=%s email=%s", user.id, user.email)
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": response.model_dump(mode="json"),
            "message": "You have successfully logged in.",
        },
    )


@user_router.post("/profile", response_model=schemas.UserProfile)
async def get_profile(current_user=Depends(get_current_user)):
    response = schemas.UserProfile(
        id=current_user.id,
        email=current_user.email,
        avatar=current_user.avatar,
        role=current_user.role,
    )
    logger.info("get_profile: user_id=%s", getattr(current_user, "id", None))
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
        user = crud.user.create(
            db,
            obj_in=schemas.UserBase(email=data.email, hashed_password=hashed_password),
        )
        logger.info("signup created: user_id=%s email=%s", user.id, email)

    otp_sent = await validate_email_send_otp(db, email, user.id)
    if otp_sent:
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
            raise HTTPException(
                status_code=status.HTTP_200_OK,
                detail="Your email is already verified.",
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
                data={"sub": user_obj.email, "id": user_obj.id}, expires_delta=refresh_expires
            )
            logger.info("verify_otp success: email=%s user_id=%s", email, user_obj.id)
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
    if not email_otp_obj:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This email is not associated with any account, please sign up.",
        )

    otp = email_otp_obj.otp
    time_now = datetime.utcnow()

    if (((time_now - email_otp_obj.updated_at).total_seconds()) / 60) < 10:
        if str(otp_sent) == str(otp):
            hashed_password = get_password_hash(new_password)
            if verify_password(new_password, user_obj.hashed_password):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="New password is same as your old password.",
                )

            user_obj = crud.user.update(
                db,
                db_obj=user_obj,
                obj_in=schemas.UserUpdate(hashed_password=hashed_password),
            )
            hashed_password = get_password_hash(new_password)

            crud.email_otp.remove(db, id=email_otp_obj.id)
            logger.info(
                "reset_forgot_password success: email=%s user_id=%s", email, user_obj.id
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

    user_obj = crud.user.update(db, db_obj=user_obj, obj_in=data)
    response = schemas.UserProfile(
        id=user_obj.id,
        email=user_obj.email,
        avatar=user_obj.avatar,
        role=user_obj.role,
    )
    logger.info("update_profile success: user_id=%s", user_obj.id)
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": response.model_dump(mode="json"),
            "message": "Profile updated successfully.",
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
