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
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, path=outfile_path)


