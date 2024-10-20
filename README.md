# Predict Maintenance by Detecting Anomalies with Photon 2 and Edge Impulse

![](/images/conv_035_comp.jpg)

Imagine if your equipment could tell you when something is wrong before it actually breaks down—saving time, reducing costs, and preventing unnecessary part replacements. That’s the power of predictive maintenance, and this tutorial shows you how to harness it with real-time anomaly detection using the Particle Photon 2 and Edge Impulse.

With traditional maintenance, schedules are set based on assumptions, such as servicing ship engines after a certain number of hours or cars after a set mileage. But sometimes, these assumptions lead to over-maintenance, replacing parts that don’t need it yet. What if instead, we could monitor equipment continuously and act only when something unusual happens—like a conveyor belt vibrating irregularly?

In this hands-on guide, you’ll learn how to build an anomaly detection system that monitors the health of a conveyor belt using the Photon 2, accelerometer sensors, and machine learning models created with Edge Impulse. You’ll also set up real-time notifications and integrate your system into IoT dashboards to visualize and act on the data.


## What is Anomaly Detection?

As the term suggests, Anomaly Detection (AD) is about detecting abnormal behavior in different scenarios, like fraud detection, quality control, early detection of failure etc. In addition to machine learning (ML), there are scores of different "traditional" AD methods: statistical methods, rule-based systems, time series analysis, pattern recognition, etc.

## This tutorial

This tutorial will show how you can perform anomaly detection of vibration in a moving conveyor belt with the Photon 2 and Edge Impulse. In addition, it demonstrates how you can build Particle integrations to notification services, and to an external IoT platform for graphical dashboards. 

### How does it work?

The end solution is demonstrated in the video below. The first half of the video shows that the conveyor belt vibrates within boundaries when transporting Particle products. In the second half of the video a non-Particle product is transported, and the belt starts to vibrate abnormally*, which is also visually indicated with a red status color! A video where you can also hear the belt struggling, is on [YouTube](https://youtu.be/TvAK7TOfutE).

![](/images/demo_800x370.gif)

* *The real reason for the anomaly was not the non-Particle product, but that I simulated a faulty gearbox by quickly changing the conveyor belt speed  back and forth*

# Table of Contents

As this tutorial is covering quite a lot of ground, the below table of contents should make it easier to grasp:

![](/images/TOC.png)


# 1. Hardware
## 1.1 Bill of Materials

- [Photon 2](https://store.particle.io/products/photon-2), or any other Particle device with SPI 

![](/images/Photon2.jpg)

- Accelerometer [ADXL362 (GY362)](https://www.amazon.com/GY-362-ADXL362-Accelerometer-Interface-Arduino/dp/B07QS9LT8J), also part of the [Particle Edge ML Kit](https://docs.particle.io/reference/datasheets/accessories/edge-ml-kit/#adxl362-gy362-accelerometer-breakout)

![](/images/adxl362-1.jpeg)

- Solderless breadboard
- Jumper wires
- USB micro-B cable
- Conveyor belt, or any other machinery you want to monitor, e.g. power drill, leaf blower, mower, fridge, vehicle, robot vacuum cleaner, etc.
    - You can also implement a [fall detection system](https://docs.edgeimpulse.com/experts/accelerometer-and-activity-projects/arduino-kway-fall-detection), similarly as did earlier with Edge Impulse's anomaly detection.


## 1.2 Assembly


Connect the Photon 2 and accelerometer like this:

| Accelerometer| Color	| Photon 2	| Details | 
| ------| ----- | ------| ------------------| 
| INT1	| 	    | Any	| Any available GPIO if using interrupt 1 (optional)| 
| INT2	| 	    | Any	| Any available GPIO if using interrupt 2 (optional)| 
| CS	| Yellow| Any	| SPI Chip Select. Use any available GPIO (required)| 
| SDO	| Green	| MISO	| SPI MISO (required)| 
| SDI	| Blue	| MOSI	| SPI MOSI (required)| 
| SCL	| Orange| SCK	| SPI SDK (required). Not I2C SCL (D1)!| 
| GND	| Black	| GND	| Ground| 
| VIN	| Red	| V3	| 3.3V power| 

You can of course use wires of any color, as long as you wire them correctly! 

Double check all connections, go grab a coffee, and check again before plugging in the USB-cable! The only smoke you want to see is from your coffee...
A tip, the text on both the accelerometer and Photon 2 is quite small, why not take a close-up photo with your mobile and use that as reference.

**Accelerometer wiring:**

![](/images/accel_10_comp.jpg)

**Photon 2 wiring:**

![](/images/accel_20_comp.jpg)

## 1.3 Activate the Photon 2

**Important!** You must register your P2 on your Particle account before continuing

Plug your P2 into your computer. Head to https://docs.particle.io/device-setup/ and follow the instructions to register it.

## 1.4 Optional: 3D-print a case

This step is optional, but if you are going to attach the accelerometer to any equipment, you need to protect the electronic somehow.
If you want to 3D-print a case and a lid, feel free to use the below STL-files:

- [Case](https://github.com/baljo/particle_anomaly_det/blob/main/images/Accel_case.stl)
- [Lid](https://github.com/baljo/particle_anomaly_det/blob/main/images/Accel_LID.stl)

I printed with white TPU, and a print quality of 100 micron (0.1 mm). To attach the case to the conveyor belt, I used double-sided tape.


![](/images/conv_040_comp.jpg)

![](/images/3D_case_010.jpg)


# 2. Build a machine learning model in Edge Impulse

This step consists of collecting vibration data from a device you want to monitor, build a machine learning (ML) model, and finally deploying it to the edge device of your choice, in this case Particle Photon 2. 

## 2.1 Get up and running ##

This [tutorial](https://docs.edgeimpulse.com/docs/edge-ai-hardware/mcu/particle-photon-2) is covering the basics of how to get machine learning running with Photon 2. If Edge Impulse is new to you, I suggest you start with following the tutorial steps, and to replicate this particular project, follow these [steps](https://docs.edgeimpulse.com/docs/tutorials/end-to-end-tutorials/continuous-motion-recognition). If you are eagerly waiting for your Particle Photon 2 or accelerometer to arrive at your door, feel free to use your [mobile phone](https://docs.edgeimpulse.com/docs/edge-ai-hardware/using-your-mobile-phone) in the meantime!

Note: The above mentioned tutorial is also using a microphone, and the accelerometer is plugged into the breadboard, but otherwise the concept is the same. 

## 2.2 Notes for this project ##

For this conveyor belt project, I collected data for various normal conditions:

| Belt moving | Speed (mm/s) | Load             |
| ---         | ---          | ---              |
| idle        | -            | -                |
| slow        | 10           | with and without |
| medium      | 20           | with and without |
| fast        | 30           | with and without |



In above cases, I collected data both when the belt was under load, and when it wasn't. As I was only interested in anomaly detection, and not classification, I did not classify the different speeds, that's why the only class I used is `normal`.

As collecting data with Particle products and Edge Impulse is very easy, I suggest you collect at least a few minutes of data already from the beginning! This way the first proof of concept will be closer to the final model.

When creating the impulse, I used a window size of 1000 ms and stride of 200 ms. These settings, as well as many others, might vary case by case. While my project was successful, there's still room for hardware or software improvements, as I sometimes got a false positive when the model detected an anomaly where I thought it shouldn't have. 

![](/images/ei_10.jpg)

## 2.3 Deploy a Particle library

When deploying a ML model to a Particle device, you have two choices:
- As a general C/C++ library: A portable C++ library with no external dependencies, which can be compiled with any modern C++ compiler.
- As a Particle library: Generates a Particle library that can be imported into a Device OS app to run on Particle development boards, SoMs, and gateways.

Unless you are a seasoned C/C++-programmer, you should choose the Particle library option.

![](/images/ei_15.jpg)


# 3. Compile the Particle firmware with Particle Workbench and Docker

## 3.1 Install the Particle Workbench

Install the Particle Workbench by [following these instructions](https://docs.particle.io/workbench/). I recommend simply installing the [VS Code extension](https://docs.particle.io/quickstart/workbench/#workbench-extension-installation).(

## 3.2 Import and compile the application

Follow these [steps](https://docs.edgeimpulse.com/docs/run-inference/running-your-impulse-particle). Especially on a Windows-computer, you might get an error message `Argument list too long`. In that case you should install and use Docker as shown later.

## 3.3 Install and compile using Docker

- Follow these [steps](https://docs.particle.io/getting-started/machine-learning/doorbell/#building-using-docker).
    - As an example, after above steps were completed, this is  the command I used in the terminal window: `docker run --name=edge-compile4 -v C:\Users\...\Dropbox\Github\Particle\particle_anomaly_det:/input -v C:\Users\...\Dropbox\Github\Particle\particle_anomaly_det:/output -e PLATFORM_ID=32 particle/buildpack-particle-firmware:5.9.0-p2`
- Flash the built firmware with this command in the terminal window (shortcut key Ctrl+J): `particle flash --local firmware.bin`.

## 3.4 Test the compiled application

To verify the concept built so far works, you should test it by typing this in the terminal window: `particle serial monitor --follow`. This should show the output of your ML model as running on the Photon 2.


# 4. Set up integrations to Pushover and Losant in the Particle console

This chapter covers how you can take the concept a bit further and get notified of anomalies through an external service, like Pushover or why not Twilio. You can select to implement just one of them, or both.
In addition, you'll learn how to update the program running on your Photon so it publishes detected anomalies as well as the current vibration status.

## 4.1 Pushover

### 4.1.1 Set up Pushover

- Create an account at Pushover (or similar service, e.g. Twilio)
  - Create an application in Pushover
  - Take a note of the User Key and API token
- Also install the Pushover app on your mobile device to get notifications


**User key field in Pushover:**

![](/images/Pushover_user_key.jpg)

**API Token/key in Pushover:**

![](/images/Pushover_API_key.jpg)



## 4.1.2 Set up a Particle Webhook to Pushover

- In the Particle console, Go to `Integrations`
- Add a new integration
- Scroll down and select `Custom Webhook`
- Select ´Custom template`
- Paste in the code below
- Replace the `token` and `user` in the code with the ones from your chosen service 
- `event` - in this case *"Anomaly score: "* - should be same event as you are publishing from your code
- Test the integration, if everything is set up correctly, you should get a notification on your mobile device
  - If you use Pushover, the notification on your mobile is received through the Pushover service, not as a SMS.



**Code to insert into the `Custom template:`**
```
{
    "name": "Anomaly score: ",
    "event": "Anomaly score: ",
    "responseTopic": "{{PARTICLE_DEVICE_ID}}/hook-response/{{PARTICLE_EVENT_NAME}}",
    "disabled": true,
    "template": "webhook",
    "url": "https://api.pushover.net/1/messages.json",
    "requestType": "POST",
    "noDefaults": true,
    "rejectUnauthorized": true,
    "unchunked": false,
    "dataUrlResponseEvent": false,
    "form": {
        "token": "API-key to selected service",
        "user": "User key to selected service",
        "title": "Anomaly score: ",
        "message": "{{{PARTICLE_EVENT_VALUE}}}"
    }
}
```

**Screenshot of the same:**

![](/images/Pushover_webhook.jpg)

## 4.2 Losant

This was the first time I used Losant, so it took me some time to understand how things connect. As guidance I used [this slighly outdated tutorial](https://www.losant.com/blog/how-to-integrate-particle-with-losant) from Losant.

### 4.2.1 Set up Losant

- Create an account at Losant
- Add a user API token, and **store it in a secure place** as you can't check it later. If you've misplaced it, you need to create a new one!  

![](/images/losant_010.jpg)

- Create an application, in my case I named it Conveyor status.

![](/images/losant_015.jpg)

- Inside the recently created application, create a webhook
- Copy the URL as you'll need it soon

![](/images/losant_020.jpg)


### 4.2.2 Set up a Particle Webhook to Losant

Creating a webhook is done similarly as for the Pushover service, the only differences are:
- Paste in the code below 
- Replace the `URL` with the one you copied from the Losant webhook
- Replace the `api_key` with the API token you created in Losant (You did store it, didn't you?)

**Code to insert into the `Custom template:`**
```
{
    "name": "Conveyor belt anomalies",
    "event": "losant_ad_score",
    "responseTopic": "{{PARTICLE_DEVICE_ID}}/hook-response/{{PARTICLE_EVENT_NAME}}",
    "disabled": false,
    "template": "losant",
    "url": "https://triggers.losant.com/webhooks/Eu3................",
    "requestType": "POST",
    "noDefaults": false,
    "rejectUnauthorized": true,
    "unchunked": false,
    "dataUrlResponseEvent": false,
    "json": true,
    "query": {
        "api_key": "Insert your very long Losant API-key here",
        "data": "{{{PARTICLE_EVENT_VALUE}}}"
    }
}
```


**Screenshot of the same:**

![](/images/Losant_webhook.jpg)


## 4.3 Publish anomalies detected through the Photon program  

The current program running on your Photon is "just" running inference and showing the results in the terminal window. To be able to get notified through Pushover, and to see the current status in Losant, you also need to publish anomalies from the program running on your Photon 2.

- Copy the program I created from [here](https://github.com/baljo/particle_anomaly_det/blob/main/src/main.cpp)
- Change the line `#include <PhotAD2_inferencing.h>` to include the header file from the program you compiled earlier.from

```
...
/* Include ----------------------------------------------------------------- */
#include "Particle.h"
#include "ADXL362DMA.h"

// This must come after Particle.h
#include <PhotAD2_inferencing.h>
...
```


- Around line 184 you'll find the code for anomaly reporting, see the code snippet below
- This line is publishing anomalies to Pushover: `Particle.publish("Anomaly score: ", String(result.anomaly), PRIVATE);` 
- This line is publishing anomalies to Losant: `Particle.publish("losant_ad_score:", String(result.anomaly), PRIVATE);` 
- As you can see, **immediate** publishing is done when the anomaly score is greater than 2.5. This threshold value depends on your equipment, on the ML model you built, on the data gathered, and on what vibration you consider to be abnormal. In my case, I did not have a controlled environment (doors slamming, cat on the table(!), etc.), so I got a few false positives.
- The user LED on your Photon 2 is also lit up when the anomaly score is over the threshold with this line `digitalWrite(D7, HIGH);`
- In addition, I wanted to monitor the vibration readings in Losant on a regular basis, basically every 10 seconds publishing the current anomaly score. This is accomplished in this section ` if ( currentTime - lastPublishTime >= publishInterval) {`
- Compile and flash the firmware as previously
- If everything is set up correctly, your mobile should get notifications through Pushover when you are slightly moving the accelerometer!

```
...
    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);

    unsigned long currentTime = millis();
    if ( currentTime - lastPublishTime >= publishInterval) {
        Particle.publish("losant_ad_score:", String(result.anomaly), PRIVATE);
        lastPublishTime = currentTime;
    }

    if (result.anomaly > 2.5) {
        Particle.publish("Anomaly score: ", String(result.anomaly), PRIVATE);
        Particle.publish("losant_ad_score:", String(result.anomaly), PRIVATE);
        digitalWrite(D7, HIGH);
    }
    else {
        digitalWrite(D7, LOW);
    }

#endif
...
```


# 5. Create a dashboard in Losant

In this final step you'll learn the basics of how to create a workflow and a dashboard in Losant. This might, at first glance seem a bit overwhelming, but once you get the first integration up and running from beginning to end, you'll find it fairly straightforward.



## 5.1 Create a device

To be able to use data from different devices, you need to create them in Losant.

- Add a new device (standalone), add `particle_device_id`as key, and paste the device id of your Photon 2 from the Particle console to the value field.

![](/images/losant_100_comp.jpg)

- Go to the Attributes tab, click on the green Add button, and add a value attribute like this:

![](/images/losant_110_comp.jpg)



## 5.2 Create a workflow

This picture shows an overview of the steps you'll need to take.

![](/images/losant_030_cropped.jpg)

- Select the earlier created application, and create a new **workflow**

![](/images/losant_040.jpg)

- Add a **webhook**

![](/images/losant_045.jpg)

- Select the **webhook you created** in the previous chapter

![](/images/losant_050_comp.jpg)

- Add a **function**

![](/images/losant_060.jpg)

- Add the following **Javascript code** to the function window. This is not strictly necessary, but the code demonstrates how you can split up text and values separated by the colon sign.

![](/images/losant_070.jpg)

- Add **Device: Get**

![](/images/losant_080.jpg)

- **Configure** the node like this

![](/images/losant_090.jpg)

- Add **Device: State**

![](/images/losant_120.jpg)

- Select the **device you created earlier** and **insert `{{ data.value  }}`** into the value field

![](/images/losant_130.jpg)


- Finally, add a **debug** node. This comes in very handy when you want to troubleshoot the integration
- **Save** the workflow

![](/images/losant_140.jpg)

## 5.3 Test the workflow

You can test the workflow - even without the hardware connected - from the Particle console. 

- Open up your Particle console
- Select the Losant integration, click on the `Test` button and type in a value which represents the anomaly score, e.g. 9.9

![](/images/Particle_010.jpg)

- In the Losant workflow, click on the Debug icon as in the picture below. Here the same 9.9 turned up, showing that the integration works!


![](/images/losant_150.jpg)

## 5.4 Add a dashboard

Finally it's time to create a dashboard! Do note that as this tutorial is more about Particle, Edge Impulse, and integrations to Pushover and Losant, it won't go into depth about Losant's features. You'll find lots of documentation on their [website](https://docs.losant.com/). 

Having said (written?) that, this is the dashboard you'll learn how to create:

![](/images/losant_170.jpg)


- Start by creating a dashboard in Losant

### 5.4.1 Create a time-series graph

- Click on Add block, and select the Time Series Graph
- Name the graph with something meaningful (hint: "Test" is not meaningful)
- Select *Historical*
- Select the device you've earlier created, as well as the attribute
- Feel free to play around with the other settings
- Save the block
- Test the workflow as demonstrated earlier, this way you can check that also the graph is working

![](/images/losant_160.jpg)

### 5.4.2 Create a visual showing status with colors

The upper right block is showing green color when the anomaly score is 2.5 or less, when over 2.5 it turns red.

- Click once again on Add block, and select the Indicator block
- The principle is similar as with the previous block, but this time you should select *Live Stream* instead of *Historical*. The reason for this is that the indicator in this case is a snapshot of the current status.
- Add this as expression `{{value-0}} > 2.5`. I've used the same value as in the program running on Photon, but they don't have to be identical. Remember that the Photon program immediately notifies of an anomaly, but also regularly sends the current vibration status.
- Select colors as per your own preferences. 

![](/images/losant_180.jpg)

### 5.4.3 Add a gauge showing vibration numerically

This is the lower right block, showing vibration status with numbers.

- Add a Gauge block, select *Live Stream*, otherwise similar steps as before.  


# Conclusions

In this tutorial, you’ve learned how to build a predictive maintenance solution using Particle Photon 2, accelerometer sensors, and machine learning models with Edge Impulse. By detecting anomalies in real time, you can monitor equipment such as conveyor belts and trigger actions when something abnormal occurs. This approach can prevent expensive breakdowns, reduce unnecessary maintenance, and extend the lifespan of machinery.

Moreover, you explored how to integrate the solution with notification services like Pushover and IoT platforms like Losant, providing you with immediate alerts and visual dashboards for monitoring. With this setup, you now have the foundation to scale this predictive maintenance solution to other equipment, allowing you to detect issues early and address them proactively.

### Kudos to *Particle* for developing fantastic hardware and for the whole eco-system, and *Edge Impulse* for developing a code-free platform for AI on the edge! 

Feel free to clone my [Github repository](https://github.com/baljo/particle_anomaly_det)!