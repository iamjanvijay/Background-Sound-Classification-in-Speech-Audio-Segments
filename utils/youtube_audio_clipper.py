import os
import subprocess
import shutil
import librosa
import argparse

audio_list = []
cached_audio = dict()


def read_audio(index):
    if index not in cached_audio:
        cached_audio[index] = librosa.core.load(
            os.path.join(args.audio_dir, audio_list[index - 1]))
    return cached_audio[index]


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser.')
    subparsers = parser.add_subparsers(dest='mode')

    # Arguments for clipping mode.
    parser_clip = subparsers.add_parser('clip')
    parser_clip.add_argument('--audio_dir', type=str, required=True)
    parser_clip.add_argument('--clipped_audio_dir', type=str, required=True)
    parser_clip.add_argument('--clip_index_init', type=int, required=True)
    args = parser.parse_args()

    if args.mode == 'clip':

        create_folder(args.audio_dir)
        create_folder(args.clipped_audio_dir)
        clip_index = args.clip_index_init

        for filename in os.listdir(args.audio_dir):
            if filename.endswith('.wav'):
                audio_list.append(filename)

        while True:
            print("1. Add new audio.")
            print("2. Clip existing audio.")
            option = int(input("Enter 1 or 2.\n"))

            if option == 1:
                video_url = str(input("Enter video URL.\n")).strip()
                download_ps = subprocess.Popen(
                    "youtube-dl -x --audio-format wav {};".format(video_url), shell=True)
                download_ps.wait()
                filenames = [
                    filename for filename in os.listdir(
                        os.getcwd()) if filename.endswith('.wav')]
                for filename in filenames:
                    file_path = os.path.join(os.getcwd(), filename)
                    shutil.move(file_path, args.audio_dir)
                    audio_list.append(filename)

            elif option == 2:
                for i, filename in enumerate(audio_list):
                    print("{}. {}.".format(i + 1, filename))
                index = int(input("Enter audio index 1-{}.\n".format(len(audio_list))))
                print ("Reading audio, please wait for some time ..")
                assert(index >= 1 and index <= len(audio_list))
                data, sampling_rate = read_audio(index)
                start_time = str(input("Enter starting clipping time.\n"))
                time = [float(dur) for dur in start_time.strip().split(':')]
                seconds = 3600 * time[0] + 60 * time[1] + time[2] + time[3] / 100.0

                end_time = str(input("Enter ending clipping time(enter 0 for single cut).\n"))
                if end_time == '0':
                    print('Keeping default end time of 4 seconds ')
                    end_seconds = seconds + 4
                else:
                    end_time = [float(dur) for dur in end_time.strip().split(':')]
                    end_seconds = 3600 * end_time[0] + 60 * end_time[1] + end_time[2] + end_time[3] / 100.0

                start_index = int(sampling_rate * seconds)
                end_index = int(sampling_rate * end_seconds)
                while True:
                    print("[{}] : Start index : {} , end index {}".format(clip_index, start_index, end_index))
                    clipped_data = data[start_index: start_index + sampling_rate * 4]
                    librosa.output.write_wav(os.path.join(args.clipped_audio_dir, "clip_{}_{}".format(
                    clip_index, audio_list[index - 1])), clipped_data, sampling_rate)
                    clip_index += 1
                    # keep a overlap of 2 seconds
                    start_index += sampling_rate * 2
                    if (start_index + sampling_rate * 4) > end_index:
                        break
            else:
                print("Invalid choice.")
