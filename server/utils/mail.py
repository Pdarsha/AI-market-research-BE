import html
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType
import markdown
from server.services.audit_logger import audit_context, AuditLogger


from server.config import settings

logger = logging.getLogger(__name__)

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
    from server.services.audit_logger import audit_context

    with audit_context(
        action="EMAIL_SEND",
        metadata={"recipient": email, "subject": "OTP for email verification"}
    ):
        message = MessageSchema(
            subject="OTP for email verification",
            recipients=[email],
            body=get_signup_template(otp),
            subtype=MessageType.html,
        )

        fm = FastMail(conf)
        await fm.send_message(message)
        return True


def is_valid_email(email: Optional[str]) -> bool:
    """Validate email address syntax."""
    if not email or not isinstance(email, str):
        return False
    email = email.strip()
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))


def get_agent_result_template(
    agent_name: str,
    result_content: str,
    execution_time: Optional[datetime] = None,
    sources: Optional[list] = None,
) -> str:
    """Render a professional branded HTML email for scheduled Agent Workflow execution results."""
    exec_time_str = (
        execution_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        if execution_time
        else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )

    clean_content = result_content or ""
    if "<<<FOLLOW_UP_QUESTIONS>>>" in clean_content:
        clean_content = clean_content.split("<<<FOLLOW_UP_QUESTIONS>>>")[0].rstrip()

    content_html = markdown.markdown(
        clean_content,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
    )

    sources_html = ""
    if sources:
        source_items = []
        for s in sources:
            name = s if isinstance(s, str) else (s.get("file_name") or s.get("title") or s.get("name") or "Source")
            source_items.append(f"<li style='margin-bottom:4px;color:#475569;'>{html.escape(str(name))}</li>")
        sources_html = f"""
        <div style="margin-top:30px;padding-top:20px;border-top:1px solid #E2E8F0;">
            <h4 style="margin:0 0 10px 0;color:#1E293B;font-size:15px;font-weight:600;">Sources & Documents Used:</h4>
            <ul style="margin:0;padding-left:20px;">
                {''.join(source_items)}
            </ul>
        </div>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Scheduled Report: {html.escape(agent_name)}</title>
<style>
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
    th, td {{ border: 1px solid #CBD5E1; padding: 8px 12px; text-align: left; }}
    th {{ background-color: #F1F5F9; font-weight: bold; }}
    pre {{ background: #F8FAFC; padding: 12px; border-radius: 6px; overflow-x: auto; border: 1px solid #E2E8F0; }}
    code {{ background: #EEF2F6; padding: 2px 4px; border-radius: 4px; font-size: 13px; }}
    blockquote {{ border-left: 4px solid #2563EB; margin: 12px 0; padding-left: 14px; color: #475569; }}
</style>
</head>
<body style="margin:0;padding:0;background:#f4f7fb;font-family:Arial,Helvetica,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f7fb;padding:30px 0;">
<tr>
<td align="center">
<table width="650" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 5px 20px rgba(0,0,0,.08);">
    <!-- Header -->
    <tr>
        <td align="center" style="background:#0F172A;padding:30px 30px;">
            <h1 style="margin:0;color:#ffffff;font-size:26px;font-weight:700;">CapitalEdge</h1>
            <p style="margin:6px 0 0;color:#CBD5E1;font-size:14px;">AI Workspace & Document Intelligence Platform</p>
        </td>
    </tr>
    <!-- Title Banner -->
    <tr>
        <td style="padding:25px 35px 15px;background:#F8FAFC;border-bottom:1px solid #E2E8F0;">
            <h2 style="margin:0 0 6px 0;color:#1E293B;font-size:20px;">🤖 Scheduled Report: {html.escape(agent_name)}</h2>
            <p style="margin:0;color:#64748B;font-size:13px;">Executed automatically on: {exec_time_str}</p>
        </td>
    </tr>
    <!-- Main Content -->
    <tr>
        <td style="padding:30px 35px;color:#334155;font-size:15px;line-height:26px;">
            {content_html}
            {sources_html}
        </td>
    </tr>
    <!-- Footer -->
    <tr>
        <td align="center" style="background:#F8FAFC;padding:20px;border-top:1px solid #E2E8F0;">
            <p style="margin:0;font-size:12px;color:#94A3B8;">© 2026 CapitalEdge. All rights reserved.</p>
            <p style="margin:4px 0 0;font-size:12px;color:#94A3B8;">Automated Scheduled Agent Delivery • Enterprise Document Intelligence</p>
        </td>
    </tr>
</table>
</td>
</tr>
</table>
</body>
</html>"""


async def send_agent_workflow_email(
    email: str,
    agent_name: str,
    result_content: str,
    execution_time: Optional[datetime] = None,
    sources: Optional[list] = None,
) -> bool:
    """Send scheduled agent workflow result email to the associated user."""
    if not is_valid_email(email):
        logger.warning(
            "[AGENT_EMAIL] Missing or invalid email address '%s' for agent '%s'. Skipping email delivery.",
            email,
            agent_name,
        )
        return False

    email = email.strip()

    with audit_context(
        action="AGENT_SCHEDULED_EMAIL_SEND",
        metadata={
            "recipient": email,
            "agent_name": agent_name,
            "subject": f"[CapitalEdge AI] Scheduled Report: {agent_name}",
        },
    ):
        try:
            subject = f"[CapitalEdge AI] Scheduled Report: {agent_name}"
            html_body = get_agent_result_template(
                agent_name=agent_name,
                result_content=result_content,
                execution_time=execution_time or datetime.now(timezone.utc),
                sources=sources,
            )

            message = MessageSchema(
                subject=subject,
                recipients=[email],
                body=html_body,
                subtype=MessageType.html,
            )

            fm = FastMail(conf)
            await fm.send_message(message)
            logger.info(
                "[AGENT_EMAIL] Successfully sent scheduled report for agent '%s' to %s",
                agent_name,
                email,
            )
            return True
        except Exception as e:
            logger.error(
                "[AGENT_EMAIL] Failed to send scheduled report email to %s for agent '%s': %s",
                email,
                agent_name,
                e,
                exc_info=True,
            )
            AuditLogger.log(
                action="AGENT_EMAIL_DELIVERY_FAILED",
                status="failed",
                metadata={"recipient": email, "agent_name": agent_name, "error": str(e)},
            )
            return False


def get_help_query_notification_template(
    ticket_id: str,
    user_name: str,
    user_email: str,
    subject: str,
    description: str,
    category: str,
) -> str:
    """HTML email template sent to company support (hello@ojasyalabs.com)."""
    safe_subject = html.escape(subject)
    safe_name = html.escape(user_name or "User")
    safe_email = html.escape(user_email)
    safe_category = html.escape(category.capitalize())
    safe_desc = html.escape(description).replace("\n", "<br/>")

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>New Support Query</title></head>
<body style="margin:0;padding:0;background:#f4f7fb;font-family:Arial,Helvetica,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f7fb;padding:30px 0;">
<tr><td align="center">
<table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,.08);">
    <tr>
        <td style="background:#0F172A;padding:25px 30px;">
            <h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:700;">CapitalEdge Support Ticket</h1>
            <p style="margin:6px 0 0;color:#94A3B8;font-size:13px;">Ticket ID: #{ticket_id}</p>
        </td>
    </tr>
    <tr>
        <td style="padding:28px 30px;">
            <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:16px;margin-bottom:20px;">
                <p style="margin:0 0 6px;font-size:13px;color:#64748B;"><strong>From:</strong> {safe_name} (&lt;{safe_email}&gt;)</p>
                <p style="margin:0 0 6px;font-size:13px;color:#64748B;"><strong>Category:</strong> {safe_category}</p>
                <p style="margin:0;font-size:14px;color:#0F172A;"><strong>Subject:</strong> {safe_subject}</p>
            </div>
            <h3 style="margin:0 0 10px;font-size:15px;color:#0F172A;">Query / Description:</h3>
            <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:16px;font-size:14px;line-height:1.6;color:#334155;">
                {safe_desc}
            </div>
        </td>
    </tr>
    <tr>
        <td align="center" style="background:#F8FAFC;padding:16px;border-top:1px solid #E2E8F0;">
            <p style="margin:0;font-size:12px;color:#94A3B8;">CapitalEdge User Inquiry • Ojasya Labs Support</p>
        </td>
    </tr>
</table>
</td></tr>
</table>
</body></html>"""


def get_help_acknowledgment_template(
    ticket_id: str,
    user_name: str,
    subject: str,
) -> str:
    """HTML email template sent to user acknowledging ticket receipt."""
    safe_name = html.escape(user_name or "there")
    safe_subject = html.escape(subject)

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>Query Received - CapitalEdge Support</title></head>
<body style="margin:0;padding:0;background:#f4f7fb;font-family:Arial,Helvetica,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f7fb;padding:30px 0;">
<tr><td align="center">
<table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,.08);">
    <tr>
        <td style="background:#0F172A;padding:25px 30px;">
            <h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:700;">We Received Your Query</h1>
            <p style="margin:6px 0 0;color:#38BDF8;font-size:13px;">Ticket #{ticket_id}</p>
        </td>
    </tr>
    <tr>
        <td style="padding:28px 30px;">
            <p style="margin:0 0 14px;font-size:15px;color:#334155;line-height:1.5;">Hi {safe_name},</p>
            <p style="margin:0 0 14px;font-size:14px;color:#475569;line-height:1.6;">
                Thank you for contacting CapitalEdge support. We have received your query regarding <strong>"{safe_subject}"</strong> and our team is actively reviewing it.
            </p>
            <p style="margin:0 0 20px;font-size:14px;color:#475569;line-height:1.6;">
                We strive to respond to all inquiries within 24 business hours. If you have additional information to provide, please reply directly to this email.
            </p>
            <div style="border-top:1px solid #E2E8F0;padding-top:16px;">
                <p style="margin:0;font-size:13px;color:#64748B;">Warm regards,<br/><strong>CapitalEdge Support Team (Ojasya Labs)</strong></p>
            </div>
        </td>
    </tr>
    <tr>
        <td align="center" style="background:#F8FAFC;padding:16px;border-top:1px solid #E2E8F0;">
            <p style="margin:0;font-size:12px;color:#94A3B8;">© 2026 CapitalEdge • <a href="https://www.ojasyalabs.com" style="color:#0284C7;text-decoration:none;">www.ojasyalabs.com</a></p>
        </td>
    </tr>
</table>
</td></tr>
</table>
</body></html>"""


async def send_help_support_email(
    user_email: str,
    user_name: str,
    subject: str,
    description: str,
    ticket_id: str,
    category: str = "general",
) -> bool:
    """Send support notification to hello@ojasyalabs.com and acknowledgment to user."""
    company_email = "hello@ojasyalabs.com"

    with audit_context(
        action="SUPPORT_HELP_EMAIL_SEND",
        metadata={
            "user_email": user_email,
            "ticket_id": ticket_id,
            "subject": subject,
            "category": category,
        },
    ):
        try:
            fm = FastMail(conf)

            # 1. Dispatch email to company designated address
            admin_subject = f"[CapitalEdge Support #{ticket_id}] {subject}"
            admin_body = get_help_query_notification_template(
                ticket_id=ticket_id,
                user_name=user_name,
                user_email=user_email,
                subject=subject,
                description=description,
                category=category,
            )
            admin_msg = MessageSchema(
                subject=admin_subject,
                recipients=[company_email],
                body=admin_body,
                subtype=MessageType.html,
            )
            await fm.send_message(admin_msg)
            logger.info("[SUPPORT_EMAIL] Forwarded support query #%s to %s", ticket_id, company_email)

            # 2. Dispatch acknowledgment email to the user if valid email
            if is_valid_email(user_email):
                user_subject = f"[CapitalEdge] Query Received: {subject} (#{ticket_id})"
                user_body = get_help_acknowledgment_template(
                    ticket_id=ticket_id,
                    user_name=user_name,
                    subject=subject,
                )
                user_msg = MessageSchema(
                    subject=user_subject,
                    recipients=[user_email],
                    body=user_body,
                    subtype=MessageType.html,
                )
                await fm.send_message(user_msg)
                logger.info("[SUPPORT_EMAIL] Sent user acknowledgment #%s to %s", ticket_id, user_email)

            return True
        except Exception as e:
            logger.error("[SUPPORT_EMAIL] Error dispatching support ticket #%s email: %s", ticket_id, e, exc_info=True)
            AuditLogger.log(
                action="SUPPORT_EMAIL_DELIVERY_FAILED",
                status="failed",
                metadata={"user_email": user_email, "ticket_id": ticket_id, "error": str(e)},
            )
            return False

