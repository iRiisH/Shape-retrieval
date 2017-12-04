JAR=core.jar:TC.jar:Jcg.jar:Jama-1.0.2.jar

all:
	javac -cp javac -cp .:$(JAR) *.java

mesh_viewer:
	javac -cp javac -cp .:$(JAR) *.java
	java  -cp javac -cp .:$(JAR) Viewer

clear:
	rm *.class
