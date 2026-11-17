#!/usr/bin/env python3
"""Non-destructive Jetson/Linux environment and camera inventory (Python 3)."""
import datetime
import glob
import grp
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPORT = HERE / 'reports' / 'hardware_report.json'


def read(path):
    try:
        return Path(path).read_text(errors='replace').replace('\x00', '').strip()
    except OSError:
        return None


def command(argv, timeout=8):
    executable = shutil.which(argv[0])
    if not executable:
        return {'available': False, 'output': None, 'error': 'Not installed or not in PATH'}
    try:
        p = subprocess.run([executable, *argv[1:]], capture_output=True, text=True, timeout=timeout, errors='replace')
        return {'available': True, 'returncode': p.returncode, 'output': p.stdout.strip(), 'error': p.stderr.strip() or None}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'available': True, 'output': None, 'error': str(exc)}


def device(path):
    p = Path(path)
    try:
        stat = p.stat()
        try:
            group = grp.getgrgid(stat.st_gid).gr_name
        except KeyError:
            group = str(stat.st_gid)
        return {'path': str(p), 'mode': oct(stat.st_mode & 0o777), 'owner_uid': stat.st_uid,
                'group': group, 'readable': os.access(p, os.R_OK), 'writable': os.access(p, os.W_OK),
                'sysfs_name': read('/sys/class/video4linux/' + p.name + '/name'),
                'sysfs_device': os.path.realpath('/sys/class/video4linux/' + p.name + '/device') if p.name.startswith('video') else None}
    except OSError as exc:
        return {'path': str(p), 'error': str(exc)}


def main():
    results = []
    def check(status, name, detail):
        results.append({'status': status, 'name': name, 'detail': detail})
        print(f'[{status}] {name}: {detail}')

    commands = {name: command(args) for name, args in {
        'uname': ['uname', '-a'], 'lsusb': ['lsusb'],
        'v4l2_list_devices': ['v4l2-ctl', '--list-devices'],
        'gst_version': ['gst-launch-1.0', '--version'],
        'gst_argus_plugin': ['gst-inspect-1.0', 'nvarguscamerasrc'],
        'argus_service': ['systemctl', 'is-active', 'nvargus-daemon'],
        'l4t_package': ['dpkg-query', '-W', '-f=${Version}', 'nvidia-l4t-core'],
        'jetpack_package': ['dpkg-query', '-W', '-f=${Version}', 'nvidia-jetpack'],
        'nvcc_version': ['nvcc', '--version'],
    }.items()}
    cameras = [device(p) for p in sorted(glob.glob('/dev/video[0-9]*'))]
    media = [device(p) for p in sorted(glob.glob('/dev/media[0-9]*'))]
    opencv = {'available': False, 'version': None, 'gstreamer_build': None, 'cuda_device_count': None}
    try:
        import cv2
        opencv.update(available=True, version=cv2.__version__)
        opencv['gstreamer_build'] = next((line.strip() for line in cv2.getBuildInformation().splitlines() if 'GStreamer:' in line), None)
        try:
            opencv['cuda_device_count'] = cv2.cuda.getCudaEnabledDeviceCount()
        except Exception as exc:
            opencv['cuda_error'] = str(exc)
    except Exception as exc:
        opencv['error'] = str(exc)

    memory = read('/proc/meminfo')
    disk = shutil.disk_usage(HERE)
    model = read('/proc/device-tree/model')
    l4t = read('/etc/nv_tegra_release')
    argus = commands['gst_argus_plugin']
    check('PASS' if model and 'Jetson' in model else 'WARN', 'Board', model or 'Device-tree model unavailable')
    check('PASS', 'Python', sys.version.split()[0] + ' (' + sys.executable + ')')
    check('PASS' if opencv['available'] else 'FAIL', 'OpenCV', opencv['version'] or opencv.get('error', 'not available'))
    check('PASS' if commands['gst_version']['available'] and commands['gst_version'].get('returncode') == 0 else 'WARN', 'GStreamer', commands['gst_version'].get('output') or 'not available')
    check('PASS' if argus.get('returncode') == 0 else 'WARN', 'Argus GStreamer plugin', 'available' if argus.get('returncode') == 0 else 'unavailable or could not inspect')
    check('PASS' if commands['lsusb'].get('returncode') == 0 else 'WARN', 'USB enumeration', 'available; see JSON' if commands['lsusb'].get('returncode') == 0 else 'unavailable or failed')
    check('PASS' if commands['v4l2_list_devices']['available'] else 'WARN', 'v4l2-ctl', 'available' if commands['v4l2_list_devices']['available'] else 'not installed; device capabilities and interface cannot be confirmed')
    if cameras:
        accessible = [d for d in cameras if d.get('readable') and d.get('writable')]
        check('PASS' if accessible else 'FAIL', 'Video device access', f'{len(cameras)} device node(s), {len(accessible)} readable and writable; capture capability and frames not yet verified')
        for d in cameras:
            check('PASS' if d.get('readable') and d.get('writable') else 'WARN', d['path'], f"name={d.get('sysfs_name') or 'unknown'}, group={d.get('group')}, read={d.get('readable')}, write={d.get('writable')}")
    else:
        check('FAIL', 'Video devices', 'No /dev/video* nodes found; no camera confirmed. CSI/Argus source is not proven absent by this inventory alone.')
    count = opencv.get('cuda_device_count')
    check('PASS' if isinstance(count, int) and count > 0 else 'WARN', 'OpenCV CUDA', f'{count} CUDA-enabled device(s) reported by OpenCV; this does not establish system-wide CUDA availability')
    check('WARN', 'Camera capture', 'Not tested in this inventory; successful device access and live frames require a separate capture test')
    report = {
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'host': {'model': model, 'platform': platform.platform(), 'os_release': read('/etc/os-release'),
                 'l4t_release': l4t, 'python_version': sys.version, 'python_executable': sys.executable,
                 'memory_info': memory, 'disk_bytes': {'total': disk.total, 'used': disk.used, 'free': disk.free},
                 'uid': os.getuid(), 'groups': [grp.getgrgid(g).gr_name for g in os.getgroups()]},
        'opencv': opencv, 'commands': commands,
        'camera': {'video_devices': cameras, 'media_devices': media,
                   'nvcsi_control': device('/dev/nvhost-ctrl-nvcsi') if Path('/dev/nvhost-ctrl-nvcsi').exists() else None,
                   'note': 'Video/media nodes and Argus plugin do not prove a usable camera; nodes may be non-capture devices.'},
        'checks': results,
        'phase_1_ready_for_capture': bool(cameras and any(d.get('readable') and d.get('writable') for d in cameras) and opencv['available']),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print(f'\nReport: {REPORT}')
    print('Next gate: verify an actual camera and capture frames before continuing.')
    return 1 if any(r['status'] == 'FAIL' for r in results) else 0


if __name__ == '__main__':
    sys.exit(main())
