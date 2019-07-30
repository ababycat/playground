color c = color(59, 68, 56);//the secret colour
String word = "Openprocessing";
String allwords ="I Have No Regrets Sensei... I Have Completed This Arduous Task Under Immense Pressure";
PVector start  =new PVector(10, 60);
int tSize =120; //Textsize
ArrayList<particle> Points = new ArrayList<particle>();
int index=0;
float restZ=0;
int F = 0;
float CTime=80;//number of frames between words
int PNum =2500;//number of particles
void setup() {
  size(1080, 720);
  frameRate(30);
  background(77, 89, 55);
  textSize(tSize);
  fill(c);
  text(word, start.x, start.y+tSize); //正在写入不可见的文本
  loadPixels(); //保存草图的所有像素
  for (int i = 0; i < PNum; i++) {//creating the particles
    Points.add(new particle(random(width), random(height)));
  }
}


void draw() {
  if (mousePressed == true) {

    background(89, 8, 78);
  } else {

    background(38, 99, 102);
  }
  int Len = word.length();
  PVector RealPix;
  if (restZ==0) {//when the timer for the word runs out
    restZ=CTime;
    for (particle P : Points) {//resetting particles and slowing them down
      P.target=false;
      P.velocity.mult(0.35);//向量乘法
    }

    String[] Arr = allwords.split(" ");
    word=Arr[F];//getting the next word
    start.x = int(random(10, width-word.length()*tSize/1.45));
    start.y = int(random(10, height-tSize*1.3));//positioning text inside the window
    fill(c);
    text(word, start.x, start.y+tSize);
    loadPixels();
    F++;
    if (F>=Arr.length) {
      F=0;
    };
  } else if (restZ<=4) {//slowing down on the last 4 frames
    for (particle P : Points) {
      P.velocity.mult(0.85);
    }
  }
  restZ-=1;
  for (int i = 0; i < 13*PNum/(CTime-30); i++) {//检查文本区域中的随机点
    RealPix=  new PVector(int(random(start.x, start.x+Len*tSize/1.45)), int(random(start.y, start.y+tSize*1.3)));
    int pixNr =int(RealPix.y*width + RealPix.x);
    color b= pixels[pixNr];
    if ((c == b)&&(restZ<CTime-20)&&(restZ>=10)) {//if the point is on text
      particle Aktuell = Points.get(index);
      if (Aktuell.target==false) {
        Aktuell.target=true;
        PVector desired = PVector.sub(RealPix, Aktuell.location);
        desired.div(restZ);
        Aktuell.velocity= desired;//kicking the particle in the direction of the point
      }
      index++;
      index=index%PNum;
    }
  }
  runP();//simulating and drawing the particles
}

void runP() {
  for (particle P : Points) {
    stroke(8, 255, 144, 128/sqrt(P.velocity.mag()+1));
    P.location.add(P.velocity);
    line(P.location.x, P.location.y, P.location.x+P.velocity.x, P.location.y+P.velocity.y);
  }//drawig particles as lines for a smoother look
}

class particle {
  PVector location;
  PVector velocity;
  boolean target=false;
  particle(float x, float y) {
    location = new PVector(x, y);
    velocity = new PVector(0.0, 0.0);
  }
}

//void setup(){
//  size(300, 300);
//}

//float noiseScale = 0.02;
//float x=0, y=0;

//float idx = 1000;
//float idy = 0;
//void draw() {
//  background(0);

//  x = noise(idx) * width;
//  y = noise(idy) * height; 

//  idx += 0.01;
//  idy += 0.01;

//  circle(x, y, 55);
//  println(x);
//}

//void mouseMoved(){
//}

//void draw(){
////line(30, 20, 85, 20);
////stroke(126);
////line(85, 20, 85, 75);
////stroke(color(255, 0, 0));
////line(85, 75, 30, 75);
//  //  pushMatrix();
//  //stroke(0);
//  //background(255);
//  //color from = color(255, 0, 0);
//  //color to = color(0, 0, 255);
//  //color interA = lerpColor(from, to, 15.1/15.2);
//  //color interB = lerpColor(from, to, 15.1/15.2);
//  //stroke(interB);
//  //line(85, 75, 30, 75);
//  //popMatrix();
//  //println(interB);





//  //fill(from);
//  //rect(10, 20, 20, 60);
//  //fill(interA);
//  //rect(30, 20, 20, 60);
//  //fill(interB);
//  //rect(50, 20, 20, 60);
//  //fill(to);
//  //rect(70, 20, 20, 60);
//}

//void draw(){
//  background(204);
//  translate(150, 150);
//  for (int i = 0; i < 10; i++) {
//    rotate(PI*0.1);
//    stroke(0);
//    float dist = abs(100);
//    line(0, 0, dist, 0);
//  }
//  println(sqrt(pow(5, 2)+));
//}

//float[] distribution = new float[360];

//void setup() {
//  size(500, 500);
//  for (int i = 0; i < distribution.length; i++) {
//    distribution[i] = int(randomGaussian() * 120);
//  }
//}

//void draw() {
//  background(204);

//  translate(width/2, width/2);

//  for (int i = 0; i < distribution.length; i++) {
//    rotate(TWO_PI/distribution.length);
//    stroke(0);
//    float dist = abs(distribution[i]);
//    line(0, 0, dist, 0);
//  }
//}
