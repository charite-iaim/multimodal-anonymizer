#!/usr/bin/env python3
"""
Konvertiert DICOM-Videos in MP4-Format.
"""

import pydicom
import numpy as np
import cv2
import sys
from pathlib import Path


def dicom_to_mp4(dicom_path: str, output_path: str = None, fps: int = 30):
    """
    Konvertiert eine DICOM-Videodatei in MP4-Format.

    Args:
        dicom_path: Pfad zur DICOM-Datei
        output_path: Pfad zur Ausgabe-MP4-Datei (optional)
        fps: Frames pro Sekunde für das Video (Standard: 30)
    """
    # DICOM-Datei laden
    try:
        dcm = pydicom.dcmread(dicom_path)
    except Exception as e:
        print(f"Fehler beim Lesen der DICOM-Datei: {e}")
        return False

    # Prüfen, ob es sich um ein Video handelt
    if not hasattr(dcm, 'NumberOfFrames') or dcm.NumberOfFrames <= 1:
        print("Dies ist kein DICOM-Video (keine oder nur ein Frame).")
        return False

    # Ausgabepfad festlegen
    if output_path is None:
        output_path = Path(dicom_path).stem + ".mp4"

    print(f"DICOM-Video gefunden mit {dcm.NumberOfFrames} Frames")

    # Pixel-Array extrahieren
    try:
        pixel_array = dcm.pixel_array
    except Exception as e:
        print(f"Fehler beim Extrahieren der Pixel-Daten: {e}")
        return False

    # Video-Dimensionen ermitteln
    if len(pixel_array.shape) == 3:
        num_frames, height, width = pixel_array.shape
        is_color = False
    elif len(pixel_array.shape) == 4:
        num_frames, height, width, channels = pixel_array.shape
        is_color = True
    else:
        print(f"Unerwartete Array-Form: {pixel_array.shape}")
        return False

    # VideoWriter initialisieren
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=is_color)

    if not out.isOpened():
        print("Fehler beim Öffnen des VideoWriters")
        return False

    # Frames konvertieren und schreiben
    print(f"Konvertiere {num_frames} Frames...")
    for i in range(num_frames):
        frame = pixel_array[i]

        # Normalisierung für Grauwert-Bilder
        if not is_color:
            # Auf 0-255 skalieren
            frame_min = frame.min()
            frame_max = frame.max()
            if frame_max > frame_min:
                frame = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
            # Zu BGR konvertieren (für OpenCV)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            # RGB zu BGR für OpenCV
            if channels == 3:
                frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                frame = frame.astype(np.uint8)

        out.write(frame)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{num_frames} Frames verarbeitet")

    out.release()
    print(f"Video erfolgreich gespeichert: {output_path}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Verwendung: python dicom_to_mp4.py <dicom_datei> [output.mp4] [fps]")
        print("Beispiel: python dicom_to_mp4.py video.dcm output.mp4 25")
        sys.exit(1)

    dicom_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    if not Path(dicom_path).exists():
        print(f"Datei nicht gefunden: {dicom_path}")
        sys.exit(1)

    success = dicom_to_mp4(dicom_path, output_path, fps)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
