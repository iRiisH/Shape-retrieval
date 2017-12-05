JAR=core.jar:TC.jar:Jcg.jar:Jama-1.0.2.jar:/usr/local/Cellar/opencv/3.3.1_1/share/OpenCV/java/opencv-331.jar

all:
	javac -cp javac -cp .:$(JAR) *.java

mesh_viewer:
	javac -cp javac -cp .:$(JAR) *.java
	java  -cp javac -cp .:$(JAR) -D$(OPENCVPATH) Viewer

feature_computer:
	javac -cp javac -cp .:$(JAR) *.java
	java  -cp javac -cp .:$(JAR) -Djava.library.path=/usr/local/Cellar/opencv3/3.3.1_1/share/OpenCV/java FeatureComputer

clear:
	rm *.class
