import datetime

def is_unix_timestamp(obj):
    try:
        # Check if it's a number (int or float)
        if isinstance(obj, (int, float)):
            # Convert to a datetime object (this will raise an error if obj is not a reasonable timestamp)
            date = datetime.datetime.fromtimestamp(obj)
            
            # Optionally, check if the date is within a reasonable range (e.g., past 1970 and not too far in the future)
            if datetime.datetime(1970, 1, 1) <= date <= datetime.datetime.now() + datetime.timedelta(days=365 * 10):
                return True
        return False
    except (OverflowError, OSError, ValueError):
        # If converting the timestamp to datetime fails or the date is unreasonable
        return False
    

def convert_unix_to_datetime(timestamp):
    if is_unix_timestamp(timestamp):
        date_time = datetime.datetime.fromtimestamp(timestamp)
        return date_time.strftime('%Y-%m-%d %H:%M:%S')
    else:
        return timestamp