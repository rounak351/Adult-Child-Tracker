import os
import subprocess
import argparse
import pytubefix
import gdown

# Set HOME variable to the directory of the current script
HOME = os.path.dirname(os.path.abspath(__file__))

def clone_and_install_repo(repo_url, requirements_file):
    repo_dir = os.path.join(HOME, 'MOT_WITH_YOLOV9_STRONG_SORT')
    if not os.path.exists(repo_dir):
        try:
            print(f"Cloning repository {repo_url}...")
            subprocess.run(['git', 'clone', '--recurse-submodules', repo_url, repo_dir], check=True)
            subprocess.run(['pip', 'install', '-r', requirements_file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error while cloning and installing repo: {e}")
            raise SystemExit(e)
    else:
        print("Repository already exists. Skipping cloning and installation.")

def download_weights(file_url, output_dir):
    output_path = os.path.join(output_dir, 'best.pt')
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Downloading YOLO weights from Google Drive to {output_path}...")
            gdown.download(file_url, output_path, quiet=False)
        except Exception as e:
            print(f"Error while downloading YOLO weights: {e}")
            raise SystemExit(e)
    else:
        print(f"YOLO weights already exist at {output_path}. Skipping download.")

def download_config_file(file_url, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Downloading configuration file from Google Drive to {output_path}...")
        gdown.download(file_url, output_path, quiet=False)
        print(f"Configuration file downloaded and replaced at {output_path}.")
    except Exception as e:
        print(f"Error while downloading configuration file: {e}")
        raise SystemExit(e)


def download_video(url, output_dir):
    try:
        print(f"Downloading video from {url}...")
        yt = pytubefix.YouTube(url)
        title = yt.title
        output_template = os.path.join(output_dir, f"{title}.mp4")
        yt.streams.get_highest_resolution().download(output_path=output_dir, filename=f"{title}.mp4")
        print(f"Downloaded video from {url} successfully as {title}.mp4!")
    except Exception as e:
        print(f"Failed to download video from {url}. Error: {e}")
        raise SystemExit(e)

def run_tracker(video_path, yolo_weights_path, strong_sort_weights_path, device):
    try:
        # Change to the directory containing the tracking script
        repo_dir = os.path.join(HOME, 'MOT_WITH_YOLOV9_STRONG_SORT')
        os.chdir(repo_dir)
        
        print(f"Running tracker on {video_path}...")
        command = [
            'python', 'trackv9.py',
            '--source', video_path,
            '--yolo-weights', yolo_weights_path,
            '--img', '640',
            '--strong-sort-weights', strong_sort_weights_path
        ]
        
        if device is not None:
            command.extend(['--device', str(device)])
        
        subprocess.run(command, check=True)
        print(f"Tracking completed for {video_path}!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run tracker on {video_path}. Error: {e}")
        raise SystemExit(e)
    finally:
        # Change back to the original directory if needed
        os.chdir(HOME)

def read_video_links(file_path):
    try:
        with open(file_path, 'r') as file:
            links = file.readlines()
        return [link.strip() for link in links if link.strip()]
    except Exception as e:
        print(f"Error while reading video links: {e}")
        raise SystemExit(e)

def main(links_file, output_dir, yolo_weights_path, strong_sort_weights_path, device):
    repo_url = 'https://github.com/TheNobody-12/MOT_WITH_YOLOV9_STRONG_SORT.git'
    requirements_file = os.path.join(HOME, 'MOT_WITH_YOLOV9_STRONG_SORT', 'requirements.txt')
    
    # New URLs for weights and config file from Google Drive
    weights_url = 'https://drive.google.com/uc?id=1BaKOAHY9dWAnvRgMqfbA1WLY1gyxRNQZ'
    weights_dir = os.path.join(HOME, 'MOT_WITH_YOLOV9_STRONG_SORT', 'weights')
    
    config_url = 'https://drive.google.com/uc?id=18c8bIkA6tZmGrstHG8O1nA7IXWJOcypA'
    config_file_path = os.path.join(HOME, 'MOT_WITH_YOLOV9_STRONG_SORT', 'strong_sort', 'configs', 'strong_sort.yaml')

    clone_and_install_repo(repo_url, requirements_file)
    download_weights(weights_url, weights_dir)
    download_config_file(config_url, config_file_path)
    
    os.makedirs(output_dir, exist_ok=True)

    video_links = read_video_links(links_file)
    
    for link in video_links:
        download_video(link, output_dir)
        video_path = os.path.join(output_dir, f"{pytubefix.YouTube(link).title}.mp4")
        run_tracker(video_path, yolo_weights_path, strong_sort_weights_path, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download YouTube videos and run tracking.')
    parser.add_argument('links_file', nargs='?', default=os.path.join(HOME, 'test_list.txt'), help='Path to a text file containing YouTube video URLs')
    parser.add_argument('--output_dir', default=os.path.join(HOME, 'Test'), help='Directory to save downloaded videos')
    parser.add_argument('--yolo_weights_path', default=os.path.join(HOME, 'MOT_WITH_YOLOV9_STRONG_SORT', 'weights', 'best.pt'), help='Path to YOLO weights')
    parser.add_argument('--strong_sort_weights_path', default=os.path.join(HOME, 'Models', 'osnet_ain_x1_0_imagenet.pt'), help='Path to StrongSORT weights')
    parser.add_argument('--device', type=int, default=None, help='Device to run tracking on (e.g., 0 for GPU or leave as None for CPU)')
    
    args = parser.parse_args()

    main(args.links_file, args.output_dir, args.yolo_weights_path, args.strong_sort_weights_path, args.device)
