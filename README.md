# audio_indices
Audio analysis for realtime field installations on limited bandwidth

## Files included in this repository:

1. `analysis.py` Contains the classifier (currently yamnet) and bioacoustic index functions
2. `audio_file.py` Runs the above analysis functions on a .wav file. Run as ```$ python3 audio_file.py wavfile.wav``` Outputs a `wavfile.npz` file (in order to match the output of `audio_stream.py`
3. `audio_stream.py` Begins a realtime audio stream, and calls `analysis.py` each time a new block of audio is buffered (blocksize set in `params.py`). This outputs a timestamp-titled zip file of numpy arrays (.npz) so as to be able to upload results using as little bandwidth as possible. The .npz file consists of two numpy arrays: *embeddings*, an array of 1024 dimensional embeddings every 0.48 seconds of audio (used for running future classifiers based on targeted research questions), and *scores_indices*, consisting of a single 525 dimensional vector for every block of audio. The first 521 entries are the yamnet classification scores (see yamnet_class_names.csv), and the last 4 are the four bioacoustic indices.
4. `params.py` Sets basic parameters for all other programs.
5. `to_csv.py` Converts .npz output to more usable formats. This saves csv files with the top 10 yamnet classes for each block of audio and with the 4 bioacoustic indices for each block of audio, and internally extracts the embeddings as a numpy array. Run as ```$ python3 to_csv.py data.npz``` and produces files named `data_classes.csv` and `data_indices.csv`
6. `yamnet_class_map.csv` Gives the class names for each class index in the yamnet scores.
7. `yamnet.tflite` The TensorFlow lite yamnet model used for the classifier.

## Sources and Additional Info

Bioacoustic indices come from https://github.com/patriceguyot/Acoustic_Indices, which borrows from the R package https://cran.r-project.org/web/packages/soundecology/vignettes/intro.html

Yamnet is documented in several places:
- https://tfhub.dev/google/yamnet/1
- https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- https://blog.tensorflow.org/2021/03/transfer-learning-for-audio-data-with-yamnet.html

---


# Full Setup and Installation Instructions
How to set up a brand new Raspberry Pi 4 to run this package (if your pi is already up and running, you can skip several of the initial steps and only install the needed libraries).

#### Installing the OS and accessing the Pi.

1. Write *Raspberry Pi OS Lite* to micro SD card, Using Raspberry Pi Imager App: https://www.raspberrypi.org/software/ 
2. Enable ssh: In a terminal window, navigate to the sd card ( e.g. type`cd /volumes/boot` on mac), then type  `touch ssh` to create an empty file with filename ssh.
3. Type`nano wpa_supplicant.conf` Then copy in the following, replacing NETWORK and PASSWORD with your wifi SSID and Password, and replacing US with your country code if outside the US, and save (`ctrl+x`) and exit:
```
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=US

network={
 ssid="NETWORK"
 psk="PASSWORD"
}
```

4. Eject the card and put in Pi, plug in the usb-c power connector, and give it ~5min to start up.

5. Connect remotely to the pi: `ssh pi@raspberrypi.local` The default password is *raspberry*, but you should follow the instructions to change it.

   **Notes:** To shut down the Pi, type `sudo halt`. Always do this before unplugging power as pulling the power directly could corrupt the sd card. To reconnect, reconnect power, give it ~1min or so to start up, then repeat step 5 with your new password.

#### Alternativey, connect over USB:

1. Do steps 1 and 2 above.
2. Type `nano config.txt` then add the line `dtoverlay=dwc2` as a new line.
3. Type `nano cmdline.txt` and add `modules-load=dwc2,g_ether` after `rootwait`.



#### Installing the required packages and programs.

1. `sudo apt-get update`
2. `sudo apt-get upgrade`
3. `sudo apt-get install python3-pip`
4. `sudo pip3 install --upgrade setuptools`
5. Shutdown `sudo shutdown`, plug in the sound card, then start the pi and reconnect via ssh as above (but with your new password).
6. Sabrent is a CM108 usb audio device. Type`sudo nano /usr/share/alsa/alsa.conf` then arrow down until you find 
   `defaults.ctl.card 0` and `defaults.pcm.card 0` and change both 0s to 1s. Save and exit. See https://cdn-learn.adafruit.com/downloads/pdf/usb-audio-cards-with-a-raspberry-pi.pdf for instructions with different usb soundcards, or follow the setup instructions for whatever ADC hat you're using. See also https://www.raspberrypi-spy.co.uk/2019/06/using-a-usb-audio-device-with-the-raspberry-pi/
7. `pip3 install scipy `Install the scipy library, which also installs numpy.
8. `sudo apt-get install libportaudio2` Install the PortAudio library.
9.  `sudo apt-get install libatlas-base-dev` Install a msudo apt-get install libportaudio2issing linear algebra library.
10.  `python3 -m pip install sounddevice `Install the Sounddevice package used to run the live audio stream.
11. `sudo apt install git` Install git, so that you can easily clone github repositories.
12. `git clone https://github.com/carneyaj/audio_indices.git` Clone the audio_indices repository containing the audio stream code, yamnet classifier, and bioacoustic indices.
13. Install the tensorflow_lite interpreter by following the Debian instructions here (4 steps): https://www.tensorflow.org/lite/guide/python
14. To see if everything is working, type `cd audio_indices/` to move to this folder, then type `python3 audio_file.py soundscape.wav` to run the classifier and bioacoustic indices on the included soundscape wave file.
15. Run `python3 audio_stream.py` to run the classifier and bioacoustic indices on a real-time audio stream. Before doing so, you can edit the parameters by typing `nano params.py` to change the recording duration, analysis block size, save directory, etc. 



#### Installing and setting up Rclone to sync data with cloud storage.

1. `curl https://rclone.org/install.sh | sudo bash`
2. `rclone config` Follow instructions here: https://rclone.org/drive/



#### Other (optional) installation notes:

1. Install circuit python then display per these instructions: https://learn.adafruit.com/adafruit-pioled-128x32-mini-oled-for-raspberry-pi/usage
2. Crontab setup: `*/5 * * * * pi python3 /home/pi/audio_indices/audio_stream.py`
3. Add crontab logging (troubleshoot): `> /home/pi/log.log 2>&1`
