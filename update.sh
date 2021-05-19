#!/bin/bash
DATE=$(date +"%Y-%m-%d_%H%M")
echo $DATE
message=$(git -C /home/pi/audio_indices pull)
echo $message
cp -r /home/pi/audio_indices/audioschedule /etc/cron.d
