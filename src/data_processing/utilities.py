from global_fun import *
import urllib.request
import tarfile

module_logger = module_logging(__file__, False)


def download_and_unzip(tar_url, outfile_path):
    """
    This is for convenience to download and unzip a tarfile
    :param tar_url: The url should end in .tar.gz
    :type tar_url: str
    :param outfile_path: where to write the tar to
    :type outfile_path: str or Path
    """
    module_logger.info(f'Downloading from: {tar_url}')
    download_location = outfile_path/Path(tar_url).name
    urllib.request.urlretrieve(tar_url, outfile_path/Path(tar_url).name)
    module_logger.info(f'Finished Download')

    with tarfile.open(download_location, 'r:*') as f:
        for file in f.getnames():
            module_logger.info(f'Extracting file: {file}')
        f.extractall(path=outfile_path)


