#!/usr/bin/env bash

set -euo pipefail

detect_loopback_device() {
  local sysfs_name
  for sysfs_name in /sys/class/video4linux/video*/name; do
    [[ -e "$sysfs_name" ]] || continue
    if grep -qi "dummy video device" "$sysfs_name"; then
      basename "$(dirname "$sysfs_name")"
      return 0
    fi
  done
  return 1
}

VIDEO_DEVICE="${VIDEO_DEVICE:-}"
if [[ -z "$VIDEO_DEVICE" ]]; then
  if detected="$(detect_loopback_device)"; then
    VIDEO_DEVICE="/dev/$detected"
  else
    VIDEO_DEVICE="/dev/video37"
  fi
fi

if [[ ! -e "$VIDEO_DEVICE" ]]; then
  echo "Missing V4L2 sink device: $VIDEO_DEVICE" >&2
  echo "Create a loopback device first, for example:" >&2
  echo "  sudo modprobe v4l2loopback video_nr=${VIDEO_DEVICE#/dev/video} card_label=scrcpy exclusive_caps=1" >&2
  exit 1
fi

exec scrcpy --v4l2-sink="$VIDEO_DEVICE" --no-video-playback --no-audio
