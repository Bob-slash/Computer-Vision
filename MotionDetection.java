import processing.video.*;
import processing.sound.*;
Capture video;
color trackColor;
PImage myImg;
int alarmNum;
import java.awt.*;
SoundFile file;

void captureEvent(Capture c) {
  myImg.copy(video, 0, 0, video.width, video.height, 0, 0, myImg.width, myImg.height);
  myImg.updatePixels();
  c.read();
}
void setup(){
  size(640, 480);
  video = new Capture(this,640,480);
  myImg = createImage(640,480, RGB);
  file = new SoundFile(this, "yt1s.com - LEGO Star Wars Sound Effect  Intruder Alert SFX.mp3");
  alarmNum = 0;
  video.start();
}

void draw(){
  myImg.loadPixels();
  video.loadPixels();
  
  loadPixels();
  for(int i = 0; i < video.height * video.width; i++){
    color vid = video.pixels[i];
    color Img = myImg.pixels[i];
    float d = dist(red(vid), green(vid), blue(vid), red(Img), green(Img), blue(Img));
    if(d >= 60) {
         pixels[i] = color(d);
    } else{
        pixels[i] = color(0);
    }
  }
  int count = 0;
  for(int i = 0; i < video.height * video.width; i++){
    //System.out.println(pixels[i]);
    if(pixels[i] != -16777216){
      count++;
    }
  }
  
  if(count < 1000){
    alarmNum = 0;
  }
  
  if(count > 10000 && alarmNum == 0){
    file.play();
    alarmNum++;
  }

  updatePixels();
}
