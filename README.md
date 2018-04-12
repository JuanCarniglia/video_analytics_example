# Video Analytics Example

This is a script (Python) file that reads frames from a Video, and anlyzes a few things:

* Detect People
* Marks Safe / Danger zones
* Detects a sliding door opening

Everytime an event (Person / Door) is generated, it sends this information via HTTP POST to an API Layer.

It is a simple example of a few things:

* HTTP Post
* uuid
* timestamping with time zone information
* Video / Camera settings fine tunning (brightness, saturation, etc.)
* Detecting and capturing mouse clicks on the windows.
* Detect an object using mean (Color mean)
* Person detection using Hog (DetectMultiScale) on a region, not the entire image
* PointPolygonTest
* Crashing detection (and inform)
* Colors on console output ! Yeah.

Enjoy.
