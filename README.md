# Air Cam

ABSTRACT

Air Cam is a smart home device that integrates verbal and visual artificial
intelligence. This paper outlines our research for a prototype that allows people
to find everyday objects that go missing in indoor environments. By integrating
a Google Voice Assistant (via Google Home Speaker), users can ask "where is
the TV remote?" These and other similar verbal commands activate an
overhead camera and object detector using deep learning models (Single Shot
Multibox Detector and Yolo) to detect where an item of interest is located in a
room. The location of the objects are then described in an intuitive manner,
allowing users to keep track of the most important items in their home. We
demonstrated the Air Cam prototype at the Cornell Tech Open Studio on
December 14, 2018. For a custom trained object, we obtained greater than 75%
accuracy. For other common pretrained objects (not custom trained), we
obtained approximately 50%, likely due to the change in camera angle of the
test images compared to the training data. With additional representative
training data, the system can be applied to small and medium size rooms, and
potentially reach a user base that is visually impaired to help them and others
find lost items.

Live Demo
https://www.youtube.com/watch?v=-oCpN1JeWj8

Report
https://drive.google.com/file/d/1eTupybiWCJAXVqUy2parv2I1H0WZ8upZ/view
