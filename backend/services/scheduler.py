"""
APScheduler jobs for daily market data fetch and inference.
"""
from apscheduler.schedulers.background import BackgroundScheduler
import pytz

def daily_inference_job():
    """
    Job that runs daily after market close.
    Calls fetcher in live mode (latest 30 days), runs inference, 
    and saves results to processed/.
    """
    pass

def start_scheduler():
    """Start the APScheduler for daily tasks."""
    scheduler = BackgroundScheduler(timezone=pytz.timezone("US/Eastern"))
    # Run at 16:30 EST (after market close)
    scheduler.add_job(daily_inference_job, "cron", day_of_week="mon-fri", hour=16, minute=30)
    scheduler.start()
