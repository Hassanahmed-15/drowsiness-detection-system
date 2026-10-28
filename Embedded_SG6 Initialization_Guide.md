# Jetson Orin Nano Initialization and USB Camera Setup

**Project:** Driver Drowsiness Monitoring, CS-477 SEECS CV Face-Off 2026  
**Subgroup:** SG-6, Embedded Deployment and System Integration

## 1. Hardware and software

| Component | Specification |
| --- | --- |
| Embedded board | NVIDIA Jetson Orin Nano 8GB |
| Camera | Generic USB Camera |
| System storage | Samsung A2 64GB microSD Card |
| Operating system | Ubuntu 22.04 through NVIDIA JetPack |
| Installation image | JetPack 6.2.1 SD card image for Jetson Orin Nano Developer Kit |
| Software update | JetPack 6.2.3, Jetson Linux 36.5.2 |
| Camera interface | USB, accessed through Linux V4L2 and Python OpenCV |

The connections below apply to the NVIDIA Developer Kit carrier board. Prepare the supplied power adapter, heatsink and fan, microSD reader, DisplayPort display, USB keyboard and mouse, Internet connection, and a Windows PC for flashing. Use a DisplayPort-to-HDMI adapter if the display only accepts HDMI.

**Installation sequence:** prepare hardware → check firmware → flash microSD → boot Ubuntu → update JetPack → install tools → connect camera → test capture.

NVIDIA specifies flashing the JetPack 6.2.1 SD image before updating to JetPack 6.2.3. [NVIDIA JetPack 6.2.3][jetpack]

## 2. Prepare the camera and microSD card

### 2.1 Generic USB Camera

1. Connect the camera to the Windows PC and open the Camera application.
2. Check that the image is clear and the cable and lens work.
3. Disconnect it after testing.

Linux compatibility will be checked on the Jetson. A UVC-compatible camera uses the generic Linux `uvcvideo` driver. The camera's supported resolution and frame rate must be read from the device rather than inferred from its description. [Linux UVC driver][uvc]

### 2.2 Samsung A2 64GB microSD Card

1. Back up any existing data. Flashing erases the card.
2. Insert the card into the PC using a microSD reader. Unlock the adapter if it has a lock switch.
3. Check that Windows detects the card at approximately its expected capacity. A nominal 64GB card may appear as about 59.6GiB.
4. Do not manually create partitions before flashing; the OS image contains the partition layout.

## 3. Check and update the board firmware

The operating system is stored on microSD, while boot firmware is stored in the board's QSPI flash. An old firmware version must be updated before starting JetPack 6.

1. Connect the DisplayPort display, USB keyboard, and supplied power adapter.
2. Press **Esc** during the NVIDIA splash screen and read the UEFI firmware version.
3. For **36.x**, proceed to Section 4.
4. For firmware **older than 36.0**, complete the update below.
5. For a different major firmware version, use NVIDIA's matching reflash instructions before installing JetPack 6.

### Update old factory firmware

1. Download the updated JetPack 5.1.3 Orin Nano image, `JP513-orin-nano-sd-card-image_b29.zip`, using the [NVIDIA firmware guide][firmware].
2. Flash it onto the Samsung card using Section 4's Etcher procedure. Boot it, complete Ubuntu setup, and connect to the Internet.
3. In the Jetson terminal, check the update service:

   ```bash
   sudo systemctl status nv-l4t-bootloader-config
   ```

4. After successful scheduling, press `q`, reboot, and wait for updating to finish:

   ```bash
   sudo reboot
   ```

5. Once JetPack 5 boots again, run:

   ```bash
   sudo nvbootctrl dump-slots-info
   sudo apt update
   sudo apt install nvidia-l4t-jetson-orin-nano-qspi-updater
   sudo reboot
   ```

6. Wait for the QSPI update to finish. The JetPack 5 image may then stop booting. Power off after updating and reflash the card with JetPack 6.2.1 in Section 4.

Keep power connected throughout firmware updating. [NVIDIA firmware update procedure][firmware]

## 4. Flash JetPack onto the microSD card

Complete these steps on the **Windows PC**.

1. Download the **JetPack 6.2.1 SD card image for Jetson Orin Nano Developer Kit** from [NVIDIA][image].
2. Download and install [balenaEtcher][etcher].
3. Insert the Samsung A2 64GB microSD card into the reader.
4. Open Etcher and select **Flash from file**.
5. Select the downloaded image ZIP. If it cannot be opened, extract it and select the `.img` file inside.
6. Select **Select target** and choose the Samsung 64GB card. Confirm its identity and capacity before continuing.
7. Select **Flash** and allow administrator access when prompted.
8. Wait for writing and validation to complete successfully.
9. Cancel any Windows prompt asking to format the flashed card. Windows may not recognize the Linux partitions.
10. Safely eject the card and remove it from the reader.

For the temporary firmware update in Section 3, use the same procedure with the JetPack 5.1.3 image instead.

## 5. Boot the Jetson and configure Ubuntu

1. With power disconnected, insert the flashed card into the microSD slot underneath the Orin Nano module.
2. Connect the display through **DisplayPort**, along with the USB keyboard, mouse, and network.
3. Check that the heatsink and fan are installed and unobstructed.
4. Connect the supplied **19V adapter to the DC power jack**. The Developer Kit powers on when power is connected.
5. If necessary, select the microSD through **Esc → Boot Manager**.
6. Complete Ubuntu setup: accept the license, select the language and keyboard layout, create an account, connect to the network, and set the timezone.
7. If prompted for the root/application partition size, use the available card capacity.
8. Open Terminal with **Ctrl+Alt+T**.

Use the DC jack for board power. The USB-C port does not provide display output. [NVIDIA hardware layout][hardware]

Run the following commands on the **Jetson**:

```bash
cat /proc/device-tree/model
cat /etc/nv_tegra_release
cat /etc/os-release
uname -m
python3 --version
findmnt /
df -h /
sudo nvbootctrl dump-slots-info
```

Before upgrading, expect Jetson Linux **36.4.4**, Ubuntu **22.04**, architecture **aarch64**, and system Python **3.10**. Confirm that the root filesystem is on the intended microSD.

Check for a scheduled firmware update and reboot after it is ready:

```bash
sudo systemctl status nv-l4t-bootloader-config
sudo reboot
```

Allow any firmware update to finish before continuing.

## 6. Update to JetPack 6.2.3

Run these steps on the fresh JetPack 6.2.1 installation.

1. Check free space and back up the NVIDIA repository configuration:

   ```bash
   df -h /
   sudo cp /etc/apt/sources.list.d/nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list.backup
   sudo nano /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
   ```

2. Change the two Jetson repository entries from `r36.4` to `r36.5`:

   ```text
   deb https://repo.download.nvidia.com/jetson/common r36.5 main
   deb https://repo.download.nvidia.com/jetson/t234 r36.5 main
   ```

3. Save with **Ctrl+O**, **Enter**, then exit with **Ctrl+X**.
4. Update packages and reboot:

   ```bash
   sudo apt update
   apt list --upgradable
   sudo apt dist-upgrade
   sudo reboot
   ```

   Review the proposed package changes before accepting. On this fresh installation, accept NVIDIA's updated configuration if prompted to replace a modified NVIDIA configuration file. Keep power connected during updating and rebooting.

5. Verify the installed release:

   ```bash
   cat /etc/nv_tegra_release
   dpkg-query -W nvidia-l4t-core
   sudo nvbootctrl dump-slots-info
   ```

The target is **Jetson Linux 36.5.2 / JetPack 6.2.3**. The repository may provide a later patch within the same branch; record the actual installed version. Use NVIDIA's package update procedure rather than Ubuntu's `do-release-upgrade`. [NVIDIA software update documentation][updates]

## 7. Install the camera tools and Python environment

All remaining commands run on the **Jetson**.

```bash
sudo apt update
sudo apt install python3-pip python3-venv python3-numpy python3-opencv v4l-utils usbutils psmisc nano git gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

Create the project folders and environment:

```bash
mkdir -p ~/drowsiness-project/embedded_sg6 ~/drowsiness-project/logs
cd ~/drowsiness-project
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -c "import cv2, numpy; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__)"
```

Use the existing team repository instead of `~/drowsiness-project` if it is already checked out. The environment uses Ubuntu's installed OpenCV and NumPy through `--system-site-packages`. Activate it in each new terminal:

```bash
cd ~/drowsiness-project
source .venv/bin/activate
```

For the project's later GPU deployment, install matching JetPack components if they are missing:

```bash
apt-cache policy nvidia-jetpack
sudo apt install nvidia-jetpack
```

Check APT's required disk space before accepting. The camera test itself does not require a GPU framework. [NVIDIA JetPack component installation][components]

## 8. Connect and configure the Generic USB Camera

### 8.1 Detect the device

1. Plug the camera directly into a **USB Type-A port** on the Jetson.
2. Run:

   ```bash
   lsusb
   v4l2-ctl --list-devices
   ls -l /dev/video*
   sudo dmesg | tail -n 30
   ```

3. Identify the capture node assigned to the camera. The commands below use `/dev/video0`; replace it if the camera uses another node.
4. Inspect its capabilities and supported modes:

   ```bash
   v4l2-ctl -d /dev/video0 --all
   v4l2-ctl -d /dev/video0 --list-formats-ext
   ```

Select a node with video-capture capability, rather than metadata only. Record one supported combination of pixel format, resolution, and frame rate. [V4L2 control utility documentation][v4l2]

### 8.2 Test streaming and preview

The following example uses **YUYV, 640 × 480, 30 FPS**. Use it only if that exact combination appears in the camera's list; otherwise substitute a listed mode in these commands and the Python test.

```bash
v4l2-ctl -d /dev/video0 --set-fmt-video=width=640,height=480,pixelformat=YUYV --set-parm=30
v4l2-ctl -d /dev/video0 --get-fmt-video --get-parm
v4l2-ctl -d /dev/video0 --stream-mmap --stream-count=120 --stream-to=/dev/null
```

Preview that mode from a terminal on the Jetson desktop. GStreamer names the YUYV format `YUY2`:

```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! 'video/x-raw,format=YUY2,width=640,height=480,framerate=30/1' ! videoconvert ! autovideosink
```

If the chosen mode is MJPG, use matching JPEG caps, for example:

```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! 'image/jpeg,width=640,height=480,framerate=30/1' ! jpegdec ! videoconvert ! autovideosink
```

Press **Ctrl+C** to stop the preview before running Python. The USB camera uses V4L2 / `v4l2src`. [NVIDIA camera interface documentation][camera]

## 9. Test camera capture in Python

Create the capture test:

```bash
cd ~/drowsiness-project
source .venv/bin/activate
nano embedded_sg6/camera_check.py
```

Paste this script and save it:

```python
import argparse
import csv
import time
from pathlib import Path

import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='/dev/video0')
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--height', type=int, default=480)
parser.add_argument('--fps', type=float, default=30)
parser.add_argument('--format', choices=['YUYV', 'MJPG'], default='YUYV')
parser.add_argument('--seconds', type=float, default=10)
parser.add_argument('--headless', action='store_true')
args = parser.parse_args()
if min(args.width, args.height, args.fps, args.seconds) <= 0:
    parser.error('Dimensions, FPS, and duration must be positive')

Path('logs').mkdir(exist_ok=True)
cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
try:
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera; check device and permissions')
    settings = [
        (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.format)),
        (cv2.CAP_PROP_FRAME_WIDTH, args.width),
        (cv2.CAP_PROP_FRAME_HEIGHT, args.height),
        (cv2.CAP_PROP_FPS, args.fps),
        (cv2.CAP_PROP_CONVERT_RGB, 1),
    ]
    for prop, value in settings:
        if not cap.set(prop, value):
            print(f'Camera did not accept property {prop}={value}')

    start = time.monotonic()
    frame_id = 0
    with open('logs/camera_timestamps.csv', 'w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['frame_id', 'timestamp_ms', 'width', 'height'])
        while time.monotonic() - start < args.seconds:
            ok, frame = cap.read()
            received = time.monotonic()
            if not ok or frame is None:
                raise RuntimeError('Camera frame read failed')
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype.name != 'uint8':
                raise RuntimeError('Expected a uint8 BGR image with three channels')
            height, width = frame.shape[:2]
            timestamp_ms = (received - start) * 1000
            if frame_id == 0:
                print('Actual frame shape:', frame.shape)
                print('Driver-reported FPS:', cap.get(cv2.CAP_PROP_FPS))
                if (width, height) != (args.width, args.height):
                    print('Actual resolution differs from the requested resolution')
                if not cv2.imwrite('logs/camera_snapshot.jpg', frame):
                    raise RuntimeError('Could not save the camera snapshot')
            writer.writerow([frame_id, round(timestamp_ms, 3), width, height])
            # Pass frame, frame_id, and timestamp_ms to SG-1 during integration.
            frame_id += 1
            if not args.headless:
                cv2.imshow('SG-6 Camera Test', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    elapsed = time.monotonic() - start
    if frame_id == 0:
        raise RuntimeError('No frames received')
    print(f'Frames: {frame_id}; measured delivery FPS: {frame_id / elapsed:.2f}')
finally:
    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
```

Run with the mode confirmed in Section 8:

```bash
python embedded_sg6/camera_check.py --device /dev/video0 --format YUYV --width 640 --height 480 --fps 30 --seconds 10
```

For a camera mode using MJPG, change `--format YUYV` to `--format MJPG` and set its listed dimensions and FPS. For capture without a graphical display, add `--headless`.

Check the preview, terminal output, `logs/camera_snapshot.jpg`, and `logs/camera_timestamps.csv`. Confirm that the actual resolution matches the selected mode and that the image has correct colors. OpenCV property requests may be adjusted by the camera driver. [OpenCV camera API][opencv]

The script measures delivered frames through the capture loop. Its timestamps mark frame receipt using a monotonic clock; they are not exposure timestamps or end-to-end alert latency. Each run replaces the two output files.

## 10. Verify initialization and prepare integration

### Stability test

Repeat the working capture command for five minutes by changing `--seconds 10` to `--seconds 300`. Use `--headless` when a preview is unnecessary. Confirm no frame-read errors, unexpected restarts, or USB disconnections occur. The script stops on capture failure; production integration must add camera-failure handling and recovery.

In another terminal, monitor the board:

```bash
sudo nvpmodel -q
tegrastats --interval 1000
```

Record the power mode, measured capture FPS, memory use, and temperature. Press **Ctrl+C** to stop monitoring. [NVIDIA tegrastats documentation][tegrastats]

### SG-1 camera input

Pass these values from SG-6's capture process to SG-1:

| Value | Format |
| --- | --- |
| `frame` | NumPy `uint8` array, shape `(height, width, 3)`, BGR channel order |
| `frame_id` | Increasing integer for each successfully captured frame |
| `timestamp_ms` | Milliseconds from a shared monotonic origin for the run |

Use a single capture owner. Convert BGR to RGB only where the receiving module requires it. Agree on the shared timestamp origin with SG-4 before integration, and use timestamps for temporal calculations.

### Save the environment record

```bash
cat /etc/nv_tegra_release > logs/jetson_release.txt
cat /etc/os-release > logs/ubuntu_release.txt
sudo nvbootctrl dump-slots-info > logs/boot_firmware.txt
dpkg-query -W > logs/system_packages.tsv
python -m pip freeze > logs/python_packages.txt
v4l2-ctl -d /dev/video0 --list-formats-ext > logs/camera_modes.txt
```

Use the selected camera node. Keep this setup procedure and the package records together; `pip freeze` alone does not capture the system-installed dependencies.

### Completion checklist

- [ ] Ubuntu boots from the Samsung A2 64GB microSD Card after reboot.
- [ ] Jetson Linux and boot firmware versions are recorded.
- [ ] OpenCV and NumPy import successfully.
- [ ] The Generic USB Camera is detected and a supported capture mode is selected.
- [ ] Preview and saved snapshot show a clear, correctly colored image.
- [ ] Actual frame dimensions and measured capture FPS are recorded.
- [ ] The five-minute capture test completes successfully.
- [ ] Camera detection and capture still work after a reboot.
- [ ] The frame, frame ID, and timestamp interface is agreed with the receiving subgroups.

## 11. Troubleshooting

| Problem | Action |
| --- | --- |
| Black screen or UEFI shell | Check DisplayPort cabling, firmware compatibility, image validation, and Boot Manager selection. |
| Windows requests card formatting | Cancel the prompt after flashing. |
| Camera missing from `lsusb` | Try another direct USB port or cable, then retest the camera on the PC. |
| USB camera detected but no video node | Inspect `sudo dmesg`; check UVC support and try `sudo modprobe uvcvideo`. |
| Several video nodes | Use `v4l2-ctl --all` and `--list-formats-ext` to find the camera's capture node. |
| Permission denied | Check device permissions. If access requires the `video` group, run `sudo usermod -aG video "$USER"`, then log out and back in. |
| Camera busy | Close other previews and camera applications. Identify the owner with `fuser -v /dev/video0`. |
| Failed reads or GStreamer negotiation | Match the requested format, resolution, and FPS to an advertised camera mode. |
| Low FPS or blurred image | Improve lighting and try a lower advertised resolution or frame rate. |
| Preview fails without a desktop | Add `--headless` to the Python command. |
| `cv2` import fails | Check `python3-opencv` installation and recreate the virtual environment with `--system-site-packages` using system Python. |
| Unexpected restart or USB disconnection | Check the supplied power adapter, cooling, USB connection, and kernel messages. |
| Insufficient storage | Move large recordings and datasets off the system card; clear downloaded package caches with `sudo apt clean`. |

## 12. References

- [NVIDIA JetPack 6.2.3 and installation route][jetpack]
- [NVIDIA JetPack 6.2.1 SD card image][image]
- [NVIDIA firmware update procedure][firmware]
- [NVIDIA Developer Kit hardware layout][hardware]
- [NVIDIA software package updates][updates]
- [NVIDIA JetPack component installation][components]
- [balenaEtcher][etcher]
- [Linux UVC driver][uvc]
- [V4L2 control utility][v4l2]
- [NVIDIA camera interfaces][camera]
- [OpenCV capture API][opencv]
- [NVIDIA tegrastats utility][tegrastats]

[jetpack]: https://developer.nvidia.com/embedded/jetpack-sdk-623
[image]: https://developer.nvidia.com/embedded/jetpack-sdk-621
[firmware]: https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/update_firmware.html
[hardware]: https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/hardware_layout.html
[updates]: https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/SoftwarePackagesAndTheUpdateMechanism.html
[components]: https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/setup_jetpack.html
[etcher]: https://etcher.balena.io/
[uvc]: https://www.ideasonboard.org/uvc/
[v4l2]: https://manpages.ubuntu.com/manpages/jammy/man1/v4l2-ctl.1.html
[camera]: https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/CameraDevelopment/CameraSoftwareDevelopmentSolution.html
[opencv]: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
[tegrastats]: https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html
