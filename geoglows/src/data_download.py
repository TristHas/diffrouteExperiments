import os, cdsapi
from queue import Queue
from threading import Thread
from natsort import natsorted
from .config import MNT_DIR

class DownloadWorker(Thread):
    """
        A worker thread that downloads data using the provided parameters.
    
        Args:
            queue (Queue): The queue from which to retrieve the download parameters.
    
        Attributes:
            queue (Queue): The queue from which to retrieve the download parameters.
    """

    def __init__(self, queue: Queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        """
            The main method of the worker thread.
            Retrieves download parameters from the queue and calls the retrieve_data function.
        """
        while True:
            client, year, month, days = self.queue.get()
            try:
                retrieve_data(client, year, month, days)
            finally:
                self.queue.task_done()

def date_to_file_name(year: int, month: int, days: list[int]) -> str:
    padded_month = str(month).zfill(2)
    padded_day_0 = str(days[0]).zfill(2)
    padded_day_1 = str(days[-1]).zfill(2)
    return f'era5_{year}{padded_month}{padded_day_0}-{year}{padded_month}{padded_day_1}.nc'

def retrieve_data(client: cdsapi.Client,
                  year: int,
                  month: int,
                  days: list[int], ) -> None:
    """
    Retrieves era5 data.

    Args:
        client (cdsapi.Client): The CDS API client.
        year (int): The year of the data.
        month (int): The month of the data.
        days (list[int]): The list of days for which data will be retrieved.

    Returns:
        None
    """
    era_dir = os.path.join(MNT_DIR, 'era5_data')
    file_name = date_to_file_name(year, month, days)
    client.retrieve(
        f'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'data_format': 'netcdf',
            'variable': 'runoff',
            'year': year,
            'month': str(month).zfill(2),
            'day': [str(day).zfill(2) for day in days],
            'time': [f'{x:02d}:00' for x in range(0, 24)],
        },
        target=os.path.join(era_dir, file_name)
    )

def prep_download(date_range):
    """
    Downloads era5 runoff data. Logs to the CloudLog class. 
    Converts hourly runoff to daily. 
    """
    era_dir = os.path.join(MNT_DIR, 'era5_data')
    os.makedirs(era_dir, exist_ok=True)
    print('connecting to CDS')
    c = cdsapi.Client()
    
    download_requests = []
    year_month_combos = {(1941, 12)}
    year_month_combos = natsorted(year_month_combos)
    
    for year, month in year_month_combos:
        download_dates = [d for d in date_range if d.year == year and d.month == month]
        days = [d.day for d in download_dates]
        expected_file_name = date_to_file_name(year, month, days)
        download_requests.append((c, year, month, days))

    return download_requests

def launch_download(download_requests, num_processes = 10):
    queue = Queue()
    for _ in range(num_processes):
        worker = DownloadWorker(queue)
        worker.daemon = True
        worker.start()
        
    for request in download_requests:
        queue.put(request)
    queue.join()
    return queue