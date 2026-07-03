from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType

from server.config import settings

conf = ConnectionConfig(
    MAIL_USERNAME=settings.MAIL_USERNAME,
    MAIL_PASSWORD=settings.MAIL_PASSWORD,
    MAIL_FROM=settings.MAIL_FROM,
    MAIL_PORT=settings.MAIL_PORT,
    MAIL_SERVER=settings.MAIL_SERVER,
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True,
)


def get_signup_template(otp):
    return f"""
            <!DOCTYPE html>
            <html>
            <head>
            <meta charset="UTF-8">
            <title></title>
            </head>

            <body style="margin:0;padding:0;background:#f4f7fb;font-family:Arial,Helvetica,sans-serif;">

            <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f7fb;padding:40px 0;">
            <tr>
            <td align="center">

            <table width="600" cellpadding="0" cellspacing="0"
            style="background:#ffffff;border-radius:12px;overflow:hidden;
            box-shadow:0 5px 20px rgba(0,0,0,.08);">

                <!-- Header -->
                <tr>
                    <td align="center"
                        style="background:#0F172A;padding:35px 30px;">

                        <h1 style="margin:0;
                                color:#ffffff;
                                font-size:30px;
                                font-weight:700;">
                            CapitalEdge
                        </h1>

                        <p style="margin:8px 0 0;
                                color:#CBD5E1;
                                font-size:15px;">
                            AI Workspace & Document Intelligence Platform
                        </p>

                    </td>
                </tr>

                <!-- Body -->
                <tr>
                    <td style="padding:45px 40px;">

                        <h2 style="margin-top:0;
                                color:#1E293B;
                                font-size:24px;">
                            Verify Your Email
                        </h2>

                        <p style="color:#475569;
                                font-size:16px;
                                line-height:28px;">

                            Welcome to <strong>CapitalEdge</strong>!

                            <br><br>

                            Thank you for creating your account.

                            Please use the One-Time Password (OTP) below to verify your email address and complete your registration.

                        </p>

                        <!-- OTP Box -->
                        <table width="100%" cellpadding="0" cellspacing="0"
                            style="margin:35px 0;">

                            <tr>
                                <td align="center">

                                    <div style="
                                        display:inline-block;
                                        padding:18px 40px;
                                        background:#EEF4FF;
                                        border:2px dashed #2563EB;
                                        border-radius:10px;
                                        font-size:34px;
                                        font-weight:bold;
                                        color:#2563EB;
                                        letter-spacing:8px;">

                                        {otp}

                                    </div>

                                </td>
                            </tr>

                        </table>

                        <p style="
                            color:#475569;
                            font-size:15px;
                            line-height:26px;">

                            This verification code will expire in
                            <strong>10 minutes</strong>.

                        </p>

                        <p style="
                            color:#475569;
                            font-size:15px;
                            line-height:26px;">

                            If you did not create a CapitalEdge account, you can safely ignore this email.

                        </p>

                        <hr style="
                            border:none;
                            border-top:1px solid #E2E8F0;
                            margin:35px 0;">

                        <p style="
                            color:#64748B;
                            font-size:14px;
                            line-height:24px;
                            margin:0;">

                            Regards,<br>
                            <strong>CapitalEdge Team</strong>

                        </p>

                    </td>
                </tr>

                <!-- Footer -->
                <tr>
                    <td align="center"
                        style="background:#F8FAFC;
                            padding:25px;
                            border-top:1px solid #E2E8F0;">

                        <p style="
                            margin:0;
                            font-size:13px;
                            color:#94A3B8;">

                            © 2026 CapitalEdge. All rights reserved.

                        </p>

                        <p style="
                            margin-top:8px;
                            font-size:12px;
                            color:#94A3B8;">

                            Secure AI Workspace • Enterprise Document Intelligence

                        </p>

                    </td>
                </tr>

            </table>

            </td>
            </tr>
            </table>

            </body>
            </html>
            """


async def send_email(email, otp):
    message = MessageSchema(
        subject="OTP for email verification",
        recipients=[email],
        body=get_signup_template(otp),
        subtype=MessageType.html,
    )

    fm = FastMail(conf)
    await fm.send_message(message)
    return True
