/* Magic Mirror
 * Module: MMM-Remote-HeartRate-Measurement
 *
 * By Josua Öhler
 * MIT Licensed.
 */


Module.register("MMM-Remote-HeartRate-Measurement", {

    // Default module config.
    defaults: {
        heartbeat: 0
    },

    // Override dom generator.
    getDom: function () {
        var wrapper = document.createElement("div");
        var title = document.createElement("div");
        var heartbeat = document.createElement("div");
        title.innerHTML = this.config.text;
        heartbeat.innerHTML = this.config.heartbeat + " bpm";
        wrapper.appendChild(title);
        wrapper.appendChild(heartbeat);
        return wrapper;
    },

    // Sends Socket Notification to node_helper to start the Python process there
    start: function () {
        this.sendSocketNotification('CONFIG', this.config);
        Log.info('Starting module: ' + this.name);

        // Schedule chart update interval.
        var self = this;
        setInterval(function () {
            self.sendNotification('UPDATECHART', true);
        }, 10000);
    },

    socketNotificationReceived: function (notification, payload) {
        if (notification === 'COUNTER') {
            Log.log(payload.toString())
            this.config.heartbeat = payload.toString()
            this.updateDom()
        }
        ;
    },

    notificationReceived: function (notification, payload, sender) {
        const self = this;
        var pirSensor = payload.config.pirSensor;

        //if (pirSensor && notification === 'USER_PRESENCE' && sender.name === 'MMM-PIR-Sensor') {
        if (notification === 'USER_PRESENCE') {
            console.log("+++++++++++++++++++++++++++++++++++PIR Sensor: True, USER_PRESENCE+++++++++++++++++++++++++++++++++++");
        }
    }

});
