# Nvidia's India Academia Connect AI Hackathon

- `Goal` → Create a model that detects any social activities that can lead to covid spread.
- `Rank` → **7th**
- `Team` → Data Pirates🏴‍☠️
- `Hackathon Details` → [Details](https://gpuhackathons.org/index.php/event/india-academia-connect-ai-hackathon)

This is the 2nd round project of Nvidia's Hackathon. Check out the 1st round project **[here](https://github.com/buggyprogrammer/Nvidia-Hindi)**

**GO TO:** [`Problem Description`](#ProblemDescription) [`Project Overview`](#Project-Overview) [`Setup`](#Setup) [`Team Member`](#Team-members)

---

<details><summary><b>Quick Summary</b></summary>

|                     |                                                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------------------- |
| **Features**        | Mask Detection, Social Distacing detection, Dashboard of violation trend, Alert Email and Annocuncement |
| **Input**           | Camera Footage                                                                                          |
| **Performance**     | ~3FPS on i3 cpu and around ~30FPS on decent GPU (not tested)                                            |
| **Technology Used** | Opencv, yolov5, resnet, tensorflow                                                                      |

</details>

## Problem Description

In recent years covid has affected our lives in a very negative way. It took the lives of many of our loved ones; many people lost their parents, children, or relatives. Not to mention our superheroes (police, doctors) who have also lost their lives while serving our country.

Covid not only took millions of lives, but it has also had an impact on many people's lifestyles and businesses; many are in severe debt, and many have had to close their doors.

To address this issue, we must first halt the spread of covid. This can be accomplished with great efficiency by utilizing AI and machine learning.

## Project Overview

In an attempt to save people's live, we tried to build a machine learning model that can detect any covid spreading activities and alert us to reduce the covid spread.

So for this problem, we imagined the scenario of the highly crowded public areas such as airports, railway stations, banks, public meetings, and parties.

<img src='files for readme\1.gif' width='600'>

<br>For finding the suspicious covid spreading activities We tried to use the camera to detect if people are following the basic advised protocols such as social-distancing, and wearing mask. Here are few samples, currently our mask detection is only working for very closed detected face, but we are working on it.

|                        Social-Distance                        |                       Mask-Detection                        |
| :-----------------------------------------------------------: | :---------------------------------------------------------: |
| <img src='files for readme\social-distance.png' width='500'>  | <img src='files for readme\mask-detected.png' width='500'>  |
| <img src='files for readme\social-distance2.png' width='500'> | <img src='files for readme\mask-detected2.png' width='500'> |

Our model measures the distance between each people with each other in a frame to detect if the social-distancing is violeted or not. It classify the people in different risk factor as red, yellow and green zone based on the number of people around them at less then given threshold (1 meter).

|              Social-Distance Sample 1              |              Social-Distance Sample 2              |
| :------------------------------------------------: | :------------------------------------------------: |
| <img src='files for readme\1test.gif' width='600'> | <img src='files for readme\3test.gif' width='600'> |

Apart from this if it detects a given number of people in red zone (high risk) then it takes the snapshot of that particular situation with all peoples position and send an email to a local authority as an alert. Also on that given situation it also make an annoucement to the public (if any speaker were connected) to remind them to maintain their social distance.

<img src='files for readme\mail.jpeg' width='400'>

## Setup

We have used yolov4 for detecting people and measuring social distance, for that we have used

- yolov4.cfg
- coco.names
- yolov4.weights (you have to download this manually from [here](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT), as github doesn't allow to upload large files)

Also, for face and mask detection we followed the Balaji Srinivasan [tutorial](https://www.youtube.com/watch?v=Ax6P93r32KU&t=835s&ab_channel=BalajiSrinivasan) and his [repository](https://github.com/balajisrinivas/Face-Mask-Detection). But as I mentioned above, this mask detector is not performing well for far away face so we will be modifying it soon.

For the libraries requirement part you can follow uploaded [requirement.txt](https://github.com/buggyprogrammer/Prevent-covid-spread/blob/master/requirements.txt)

## Team Members

**`Data Pirates🏴‍☠️ (Our Team Name)`**
|Members|
|:-|
|[Animesh Singh](https://github.com/animeshdebug7) (me)|
|[Aishwarya Kshirsagar](https://github.com/AishwaryaKshirsagar)|
|[Aman Kumar Verma](https://github.com/buggyprogrammer)|
