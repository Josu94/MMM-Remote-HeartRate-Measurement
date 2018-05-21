/**
 * Created by josuaohler on 04.05.18.
 */

'use strict';
var NodeHelper = require('node_helper');
var spawn = require("child_process").spawn;
var pythonStarted = false


module.exports = NodeHelper.create({

    socketNotificationToModul: function (name, data) {
        const self = this;
        self.sendSocketNotification(name, data);
    },

    start_facedetection: function () {
        const self = this;

        //var childProcess = spawn('python', ["-u", "modules/MMM-Remote-HeartRate-Measurement/counter.py"], {stdio: 'pipe'});
        const options = {
            stdio: 'pipe',
        };

        var childProcess = spawn('python',
            ["-u", "modules/MMM-Remote-HeartRate-Measurement/python/faceDetection1.py",
                "-p", "modules/MMM-Remote-HeartRate-Measurement/shape_predictor_68_face_landmarks.dat"
                , self.config.piCamera], options);

        childProcess.stdout.on('data', (data) => {
            console.log(`${data}`)
            self.socketNotificationToModul('COUNTER', `${data}`);
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

        // setTimeout(function () {
        //     childProcess.stdin.write('Test stdin Wert');
        //     childProcess.stdin.end();
        // }, 5000);
    },

    start_ultrasonic_sensor: function () {
        const self = this;
        var counter = 0;

        var childProcess = spawn('python', ["-u", "modules/MMM-Remote-HeartRate-Measurement/python/ultrasonicSensorTest.py"], {stdio: 'pipe'});

        childProcess.stdout.on('data', (data) => {
            console.log(`${data}`)
            if (data > 100) {
                // Sends ultra sonic info to main modul to display it on the mirror
                counter = 0;
                self.socketNotificationToModul('US_INFO', 'Come closer to measure your Heart Rate...');
            } else {
                counter += 1;
                if (counter === 3) {
                    counter = 0;
                    // Sends ultra sonic info to main modul to display it on the mirror
                    self.socketNotificationToModul('US_INFO', 'Checking for faces...');
                    console.log('Starting face detection.');
                    self.start_facedetection();
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
        if (notification === 'START_FACE_DETECTION') {
            this.config = payload
            if (!pythonStarted) {
                pythonStarted = true;
                this.start_facedetection();
            }
        }

        else if (notification === 'CHECK_DISTANCE_ULTRASONIC_SENSOR') {
            this.config = payload
            if (!pythonStarted) {
                pythonStarted = true;
                this.start_ultrasonic_sensor();
            }
        }
    }

});
