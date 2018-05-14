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

    python_start: function () {
        const self = this;

        //var childProcess = spawn('python', ["-u", "modules/MMM-Remote-HeartRate-Measurement/counter.py"], {stdio: 'pipe'});
        var childProcess = spawn('python', ["-u", "modules/MMM-Remote-HeartRate-Measurement/ultrasonicSensorTest.py"], {stdio: 'pipe'});

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

        setTimeout(function () {
            childProcess.stdin.write('Test stdin Wert');
            childProcess.stdin.end();
        }, 5000);
    },

    // Subclass socketNotificationReceived received.
    socketNotificationReceived: function (notification, payload) {
        if (notification === 'CONFIG') {
            this.config = payload
            if (!pythonStarted) {
                pythonStarted = true;
                this.python_start();
            }
            ;
        }
        ;
    }

});