#!/bin/bash
DATE=$(date +"%Y-%m-%d_%H%M")
echo $DATE
git -C /home/pi/audio_indices pull -v
cp -r /home/pi/audio_indices/audioschedule /etc/cron.d
