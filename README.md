# Detecting Anomalies in a Conveyor Belt with Photon 2 and Edge Impulse

![](/images/conv_010.jpg)

# Problem Statement

**????????????????**

## What is Anomaly Detection?

As the term suggests, Anomaly Detection (AD) is about detecting abnormal behavior in different scenarios, like fraud detection, quality control, early detection of failure etc. In addition to machine learning (ML), there are scores of different "traditional" AD methods: statistical methods, rule-based systems, time series analysis, pattern recognition, etc.

## This tutorial

This tutorial will show how you can perform anomaly detection of vibration in a moving conveyor belt with the Photon 2 and Edge Impulse. In addition, it demonstrates how you can build Particle integrations to notification services, and to an external IoT platform for graphical dashboards. 

### How does it work?

The end solution is demonstrated in the video below. The first half of the video shows that the conveyor belt vibrates within boundaries when transporting Particle products. In the second half of the video a non-Particle product is transported, and the belt starts to vibrate abnormally*, which is also visually indicated with a red status color! A video where you can also hear the belt struggling is on [YouTube](https://youtu.be/TvAK7TOfutE).

![](/images/demo_800x370.gif)

* *The real reason for the anomaly was not the non-Particle product, but that I simulated a faulty gearbox by quickly changing the conveyor belt speed  back and forth*

# Table of Contents

This tutorial is covering quite a lot of ground, the main steps are:

    1. Setting up the hardware
        1.1 Bill of Materials

2. Building a machine learning model in Edge Impulse
3. Compiling the Particle firmware with Particle Workbench and Docker
4. Setting up integrations to Pushover and Losant in the Particle console
5. Building a dashboard in Losant

# 1. Hardware
## 1.1 Bill of Materials

- [Photon 2](https://store.particle.io/products/photon-2), or any other Particle device with SPI 

![](/images/Photon2.jpg)

- Accelerometer [ADXL362 (GY362)](https://www.amazon.com/GY-362-ADXL362-Accelerometer-Interface-Arduino/dp/B07QS9LT8J), also part of the [Particle Edge ML Kit](https://docs.particle.io/reference/datasheets/accessories/edge-ml-kit/#adxl362-gy362-accelerometer-breakout)

![](/images/adxl362-1.jpeg)

- Solderless breadboard
- Jumper wires
- USB micro-B cable
- Conveyor belt, or any other machinery you want to monitor, e.g. power drill, leaf blower, mower, fridge, vehicle, etc.

| a | b | c |
| --- | --- | --- |
| 1 | 2 | 3 |


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



# Edge Impulse

The earlier mentioned [tutorial](https://docs.edgeimpulse.com/docs/edge-ai-hardware/mcu/particle-photon-2) uses both an accelerometer as well as a microphone, but if you just want to connect the accelerometer you can connect like this:


Important! You must register your P2 on your Particle account before continuing

Plug your P2 into your computer. Head to https://docs.particle.io/device-setup/ and follow the instructions to register it.

## Get up and running ##

This [tutorial](https://docs.edgeimpulse.com/docs/edge-ai-hardware/mcu/particle-photon-2) is covering the basics of how to get machine learning running with Photon 2. If Edge Impulse is new to you, I suggest you start with following the tutorial steps, and to replicate this particular project, follow these [steps](https://docs.edgeimpulse.com/docs/tutorials/end-to-end-tutorials/continuous-motion-recognition). If you are eagerly waiting for your Particle Photon 2 or accelerometer to arrive at your door, feel free to use your [mobile phone](https://docs.edgeimpulse.com/docs/edge-ai-hardware/using-your-mobile-phone) in the meantime!

## Notes for this project ##

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



## To use in Particle Workbench

1. In Workbench, select **Particle: Import Project** and select the `project.properties` file in the directory that you just downloaded and extracted.

1. Use **Particle: Configure Project for Device** and select **deviceOS@5.3.2** and choose a target. (e.g. **P2** , this option is also used for the Photon 2).

1. Compile with  **Particle: Compile application (local)**

1. Flash with **Particle: Flash application (local)**


> At this time you cannot use the **Particle: Cloud Compile** or **Particle: Cloud Flash** options; local compilation is required.

## Examples

`src/main.cpp` already contains one of the examples found in `examples/` directory.  If
you wish to use a different example, copy the `main.cpp` file from the example
in the `examples/` directory and replace `src/main.cpp`.

You may need to install additional libraries using **Particle: Install Library**
for some examples. See the example source for details.

### static_buffer

The `static_buffer` example can be used to feed raw features directly for
inference on target.  Copy raw features from the **Live classifiaction** page of
your project into the `features` array. For more information see
https://docs.edgeimpulse.com/docs/deployment/running-your-impulse-particle.
