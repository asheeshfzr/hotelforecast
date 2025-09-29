import logging
import smtplib
from email.message import EmailMessage

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

from .config import ALERT_SLACK_WEBHOOK, ALERT_EMAIL, SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASS

logger = logging.getLogger(__name__)


def send_slack_alert(text: str):
    if not ALERT_SLACK_WEBHOOK:
        logger.info("No SLACK_WEBHOOK configured; skipping slack alert. Message: %s", text)
        return False
    if not HAS_REQUESTS:
        logger.info("requests not installed; skipping Slack alert. Message: %s", text)
        return False
    try:
        resp = requests.post(ALERT_SLACK_WEBHOOK, json={"text": text}, timeout=10)
        if resp.status_code in (200, 201, 204):
            logger.info("Sent Slack alert")
            return True
        logger.warning("Slack webhook returned %s: %s", resp.status_code, resp.text)
        return False
    except Exception as e:
        logger.exception("Failed to send Slack alert: %s", e)
        return False


def send_email_alert(subject: str, body: str):
    if not (ALERT_EMAIL and SMTP_SERVER and SMTP_USER and SMTP_PASS):
        logger.info("Email alert not configured; skipping. Subject: %s", subject)
        return False
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = SMTP_USER
        msg['To'] = ALERT_EMAIL
        msg.set_content(body)
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        logger.info("Sent email alert to %s", ALERT_EMAIL)
        return True
    except Exception as e:
        logger.exception("Failed to send email alert: %s", e)
        return False
