# Exported Edge Impulse's Particle Library for PhotAD2_inferencing


| a | b | c |
| --- | --- | --- |
| 1 | 2 | 3 |


![](/images/demo_800x370.gif)

# Edge Impulse

## Get up and running ##

This [tutorial](https://docs.edgeimpulse.com/docs/edge-ai-hardware/mcu/particle-photon-2) is covering the basics of how to get machine learning running with Photon 2. If Edge Impulse is new to you, I suggest you start with following the tutorial steps, and to replicate this particular project, follow these [steps](https://docs.edgeimpulse.com/docs/tutorials/end-to-end-tutorials/continuous-motion-recognition). If you are eagerly waiting for your Particle Photon 2 or accelerometer to arrive at your door, feel free to use your [mobile phone](https://docs.edgeimpulse.com/docs/edge-ai-hardware/using-your-mobile-phone) instead!

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


## Hardware Connections

The earlier mentioned [tutorial](https://docs.edgeimpulse.com/docs/edge-ai-hardware/mcu/particle-photon-2) uses both an accelerometer as well as a microphone, but if you just want to connect the accelerometer you can connect like this:




| Breakout	    | Color	| Connect To	| Details | 
| ------| ----- | ------| ------------------| 
| INT1	| 	    | Any	| Any available GPIO if using interrupt 1 (optional)| 
| INT2	| 	    | Any	| Any available GPIO if using interrupt 2 (optional)| 
| CS	| Yellow| Any	| SPI Chip Select. Use any available GPIO (required)| 
| SDO	| Green	| MISO	| SPI MISO (required)| 
| SDI	| Blue	| MOSI	| SPI MOSI (required)| 
| SCL	| Orange| SCK	| SPI SDK (required). Not I2C SCL (D1)!| 
| GND	| Black	| GND	| Ground| 
| VIN	| Red	| V3	| 3.3V power| 

![](/images/accel_10_comp.jpg)

![](/images/accel_20_comp.jpg)



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
