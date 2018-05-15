#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Entfernungsmesser für RaspberryPi
# Version 1.0
#
# Copyright: tfnApps.de Jens Dutzi
# Datum: 27.06.2015
# Lizenz: MIT Lizenz (siehe LICENSE)
# -----------------------------------


# import required modules
import time
import RPi.GPIO as GPIO
import faceDetection

# define GPIO pins
GPIOTrigger = 25
GPIOEcho    = 24

# Funktion zum messen der Entfernung
def MesseDistanz():
    # Trigger auf "high" setzen (Signal senden)
    GPIO.output(GPIOTrigger, True)

    # Signal für 10µs senden (ggf. je nach RaspberryPi auf 0.0005 setzen)
    time.sleep(0.00001)

    # Trigger auf "low setzen (Signal beenden)
    GPIO.output(GPIOTrigger, False)

    # Aktuelle Zeit setzen
    StartZeit = time.time()
    StopZeit = StartZeit

    # Warte bis "Echo" auf "low" gesetzt wird und setze danach Start-Zeit erneut
    while GPIO.input(GPIOEcho) == 0:
        StartZeit = time.time()

    # Warte bis "Echo" auf "high" wechselt (Signal wird empfangen) und setze End-Zeit
    while GPIO.input(GPIOEcho) == 1:
        StopZeit = time.time()

    # Abstand anhand der Signal-Laufzeit berechnen
    # Schallgeschwindigkeit: 343,50 m/s (bei 20°C Lufttemperatur)
    # Formel: /Signallaufzeit in Sekunden * Schallgeschwindigket in cm/s) / 2 (wg. Hin- und Rückweg des Signals)
    SignalLaufzeit = StopZeit - StartZeit
    Distanz = (SignalLaufzeit/2) * 34350

    return [Distanz, (SignalLaufzeit*1000/2)]

# main function
def main():
    try:
        while True:
            Ergebnis = MesseDistanz()

            if Ergebnis[0] > 100:
                print("Come closer to measure your pulse...")
            elif Ergebnis[0] < 100:
                print("Searching for faces to measure pulse...")
                faceFound = faceDetection.searchFaceViolaJones()
                if faceFound:
                    print("Face was found!")
                else:
                    print("No face was found!")
            time.sleep(1)

    # reset GPIO settings if user pressed Ctrl+C
    except KeyboardInterrupt:
        print("Messung abgebrochen")
        GPIO.cleanup()

if __name__ == '__main__':
    # benutze GPIO Pin Nummerierung-Standard (Broadcom SOC channel)
    GPIO.setmode(GPIO.BCM)

    # Initialisiere GPIO Ports
    GPIO.setup(GPIOTrigger, GPIO.OUT)
    GPIO.setup(GPIOEcho, GPIO.IN)

    # Setze GPIO Trigger auf false
    GPIO.output(GPIOTrigger, False)

    # Main-Funktion starten
    main()