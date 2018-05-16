/* Magic Mirror
 * Module: MMM-Remote-HeartRate-Measurement
 *
 * By Josua Öhler
 * MIT Licensed.
 */

Module.register("MMM-Remote-HeartRate-Measurement", {

    // Default module config.
    defaults: {
        textField: 0,
        pirSensor: false,
        ultrasonicSensor: false
    },

    // Override dom generator.
    getDom: function () {
        var wrapper = document.createElement("div");
        var title = document.createElement("div");
        var heartbeat = document.createElement("div");
        title.innerHTML = this.config.text;
        heartbeat.innerHTML = this.config.textField;
        wrapper.appendChild(title);
        wrapper.appendChild(heartbeat);
        return wrapper;
    },

    // Sends Socket Notification to node_helper to start the Python process there
    start: function () {
        var self = this;
        if (self.config.ultrasonicSensor === false) {
            this.sendSocketNotification('START_FACE_DETECTION', this.config);
        }
        Log.info('Starting module: ' + this.name);

        // Schedule chart update interval.
        // var self = this;
        // setInterval(function () {
        //     self.sendNotification('UPDATECHART', true);
        // }, 10000);
    },

    socketNotificationReceived: function (notification, payload) {
        if (notification === 'US_INFO') {
            Log.log(payload.toString())
            this.config.textField = payload.toString()
            this.updateDom()
        }
        ;
    },

    notificationReceived: function (notification, payload, sender) {
        var self = this;
        if (self.config.ultrasonicSensor && notification === 'USER_PRESENCE' && sender.name === 'MMM-PIR-Sensor') {
            //TODO: Face Detection mit Python Code über node_helper aufrufen. -->  this.sendSocketNotification('xxx', this.config);
            //TODO: Ultraschallsensor mit Python ansteuern
             this.sendSocketNotification('CHECK_DISTANCE_ULTRASONIC_SENSOR', this.config);
        }
    }

});
