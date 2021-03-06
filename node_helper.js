/**
 * Created by josuaohler on 04.05.18.
 */

'use strict';
var NodeHelper = require('node_helper');
var spawn = require("child_process").spawn;
//var spawnSync = require("child_process").spawnSync;
var python_us_started = false;
var python_us_pid = null;
var python_fd_started = false;
var python_fd_pid = null;


module.exports = NodeHelper.create({

    socketNotificationToModul: function (name, data) {
        const self = this;
        self.sendSocketNotification(name, data);
    },

    start_rapidCapture: function () {
        const self = this;

        const options = {
            stdio: 'pipe'
        }

        // TODO: Starting rapidCaptureAndProcessing. For first prototype save pictures for 60 seconds with 30fps --> 1800 pictures
        // TODO: Add option to config recording/pulse meassuring time manually
        // childProcess.stdin.write(JSON.stringify(self.config.piCamera));
        // childProcess.stdin.end();

        // Starting python process to save pictures from camera stream to Raspi SD-Card
        var childProcess = spawn('python3',
            ["-u", "modules/MMM-Remote-HeartRate-Measurement/python/facial_landmark_multiprocessing.py"], options);

        childProcess.stdout.on('data', (data) => {
            console.log(`${data}`)
            self.socketNotificationToModul('FD_INFO', `${data}`);
        });

        childProcess.on('close', (code) => {
            console.log(`child process faceDetection exited with code ${code}`);
            console.log('[INFO] Stopped HR measurement.')
            self.socketNotificationToModul('FD_INFO', 'Completed HR measurement.');
            // Starting ultrasonic sensor script again so check if a person is in front of the window
            //TODO: Face Detection abgeschlossen --> Mit Ultraschallsensor erneut prüfen, ob sich eine Person vor dem Spiegel befindet.
            //this.start_ultrasonic_sensor();
        });

        childProcess.on('exit', function (code, signal) {
            console.log('child process exited with ' +
                `code ${code} and signal ${signal}`);
        });

        childProcess.stderr.on('data', (data) => {
            console.log(`stderr: ${data}`);
        });
    },

    // start_facedetection: function () {
    //     const self = this;
    //
    //     const options = {
    //         stdio: 'pipe',
    //         // input: JSON.stringify(self.config.piCamera)      // use this option if you want to send data to a spawnSync child_process
    //     }
    //
    //     if (python_us_pid != null) {
    //         process.kill(python_us_pid);
    //         console.log("Process with PID: " + python_us_pid + " was killed.");
    //         python_us_pid = null;
    //     }
    //
    //     // Starting rapidCaptureAndProcessing. For first prototype save pictures for 60 seconds with 30fps --> 1800 pictures
    //     this.start_rapidCapture()
    //
    //     // TODO: Change existing programm (faceDetection1.py) to load pictures from SD-Card, and not directly from live stream!
    //     // Starting python process to recognise faces and detect facial landmarks
    //     var childProcess = spawn('python',
    //         ["-u", "modules/MMM-Remote-HeartRate-Measurement/python/faceDetection1.py",
    //             "-p", "modules/MMM-Remote-HeartRate-Measurement/shape_predictor_68_face_landmarks.dat"], options);
    //
    //     // Some loging and find out the pid of the spawned process
    //     python_fd_started = true;
    //     console.log(`Spawned python script "facedetection" pid: ${childProcess.pid}`);
    //     python_fd_pid = childProcess.pid;
    //
    //     // Send info to python script, if a piCamera is used or not
    //     childProcess.stdin.write(JSON.stringify(self.config.piCamera));
    //     childProcess.stdin.end();
    //
    //     childProcess.stdout.on('data', (data) => {
    //         // console.log('**************')
    //         // console.log(`${data}`)
    //         var obj = JSON.parse(data);
    //         if (obj.FACE_FOUND) {
    //             console.log(obj.FACE_FOUND + ' Face was found.');
    //             self.socketNotificationToModul('FD_INFO', obj.FACE_FOUND + ' Face was found.');
    //         } else if (obj.FPS) {
    //             console.log('FPS: ' + obj.FPS);
    //             self.socketNotificationToModul('FD_INFO_FPS', 'FPS: ' + obj.FPS);
    //         }
    //     });
    //
    //     childProcess.on('close', (code) => {
    //         console.log(`child process faceDetection exited with code ${code}`);
    //         //TODO: Face Detection abgeschlossen --> Mit Ultraschallsensor erneut prüfen, ob sich eine Person vor dem Spiegel befindet.
    //         console.log('[INFO] Stopped face detection.')
    //         self.socketNotificationToModul('FD_INFO', 'Stopped face detection.');
    //         // no Ultrasnic Sensor
    //         if (self.config.ultrasonicSensor === false) {
    //             this.start_facedetection();
    //         }
    //     });
    //
    //     childProcess.on('exit', function (code, signal) {
    //         console.log('child process exited with ' +
    //             `code ${code} and signal ${signal}`);
    //     });
    //
    //     childProcess.stderr.on('data', (data) => {
    //         console.log(`stderr: ${data}`);
    //     });
    //
    //     // setTimeout(function () {
    //     //     childProcess.stdin.write('Test stdin Wert');
    //     //     childProcess.stdin.end();
    //     // }, 5000);
    // },

    start_ultrasonic_sensor: function () {
        const self = this;
        var counter = 0;

        var childProcess = spawn('python', ["-u", "modules/MMM-Remote-HeartRate-Measurement/python/ultrasonicSensorTest.py"], {stdio: 'pipe'});
        python_us_started = true;
        console.log(`Spawned python script "ultrasonicSensor" pid: ${childProcess.pid}`);
        python_us_pid = childProcess.pid;

        childProcess.stdout.on('data', (data) => {
            console.log(`${data}`)
            if (data > 100) {
                // Sends ultra sonic info to main modul to display it on the mirror
                counter = 0;
                self.socketNotificationToModul('US_INFO', 'Come closer to measure your Heart Rate...');
            } else {
                counter += 1;
                if (counter === 3) {
                    // Sends ultra sonic info to main modul to display it on the mirror
                    self.socketNotificationToModul('US_INFO', 'Start measuring heartrate...');
                    // console.log('Starting face detection.');
                    // self.start_facedetection();
                    console.log('Start measuring heartrate...');
                    self.start_rapidCapture();
                    console.log('Next: Kill ultrasonic sensor python child process.')
                    childProcess.kill();
                }
            }
        });

        childProcess.on('close', (code) => {
            console.log(`child process exited with code ${code}`);
            // process.exit();
        });

        childProcess.on('exit', function (code, signal) {
            console.log('child process exited with ' +
                `code ${code} and signal ${signal}`);
        });

        childProcess.stderr.on('data', (data) => {
            console.log(`stderr: ${data}`);
        });
    },

    // Subclass socketNotificationReceived received.
    socketNotificationReceived: function (notification, payload) {
        if (notification === 'START_RAPID_CAPTURE') {
            this.config = payload
            if (!python_us_started) {
                this.start_rapidCapture();
            }
        }

        else if (notification === 'CHECK_DISTANCE_ULTRASONIC_SENSOR') {
            this.config = payload
            if (!python_us_started) {
                this.start_ultrasonic_sensor();
            }
        }
    }

});
