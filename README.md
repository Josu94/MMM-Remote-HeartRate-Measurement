# MMM-Remote-HeartRate-Measurement
Remote Photoplethysmography Module for MagicMirror² to remotly meassure pulserate via webcam or raspicam.

# Information
This module meassure your pulsrate remotely via a normal RGB consumer camera. For best UX, additional sensors are required. 

- PIR Sensor: HC-SR501  
- Ultrasonic Sensor: HC-SR04

If you don't want to use these additional sensors, you can still use this model without these sensors. Just setup the config.js file.

# Dependencies
- An instalation of [MagicMirror²](https://github.com/MichMich/MagicMirror) 
- python3
- pip
- imutils
- matplotlib
- scipy
- dlib
- cv2

# Installation
1. Run these commands from the root directory of your magic mirror installation. `cd modules` `git clone https://github.com/Josu94/MMM-Remote-HeartRate-Measurement.git` 
2. Run command `dependencies.sh` in `~/MagicMirror/modules/MMM-Remote-HeartRate-Measurement/installers` directory, to install all dependencies. This will need a couple of minutes.
3. Configure your `~/MagicMirror/config/config.js`:
```
{
    module: 'MMM-Remote-HeartRate-Measurement',
    position: 'middle_center',
    config: {
        text: "Remote-HeartRate-Measurement",
        pirSensor: false,
        ultrasonicSensor: true,
        piCamera: true
    }
}
```

# Configuration options
| Option              | Default                             | Description                                                     |
| :-------------      | :----------------                   | :-------------                                                  |
| `text`              | MMM-Remote-HeartRate-Measurement    | The displayed name for this module in the middle of the screen. |
| `pirSensor`         | false                               | Define if a pirSensor on port TODO is connected to the pi.         |
| `ultrasonicSensor`  | false                               | Define if a ultrasonicSensor on port TODO is connected to the pi.  |
| `piCamera`          | false                               | Define if a pi camera is beeing used.          |
| `distance`          | TODO                                | TODO          |
| `time`              | TODO                                | TODO          |

# Usage
If you are using a pirSensor, the `MMM-Remote-HeartRate-Measurement` Module will be informed about a detected motion from the TODO module. 
After that, the ultrasonicSensor is activated and checks, if the distance between smart mirror and the person in front of it is smaller than the configured `distance` value.
If this distance is dropping below the minimum value, the main python script is being executed. The first bpm value will be displayed after 30s. After that it will be updated 
in an one second interval for the defined `time`. When the time is up, the average heart rate will be displayed and the python script terminates. 
