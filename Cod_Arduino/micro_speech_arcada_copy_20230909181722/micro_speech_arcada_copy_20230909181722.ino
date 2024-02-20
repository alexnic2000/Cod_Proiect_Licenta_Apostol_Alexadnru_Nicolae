/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <string.h>
#include <Adafruit_NeoPixel.h>
#include <CircularBuffer.h>
#include <Wire.h>
#include <SPI.h>
#include <Adafruit_LIS3DH.h>
#include <Adafruit_Sensor.h>



#include <TensorFlowLite.h>
#include "Adafruit_Arcada.h"
Adafruit_Arcada arcada;

#include "main_functions.h"
#include "third_party/kissfft/_kiss_fft_guts.h"
#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_features_micro_model_settings.h"
#include "micro_features_tiny_conv_micro_features_model_data.h"
#include "recognize_commands.h"
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Pinii accelerometrului
#define LIS3DH_CLK 13
#define LIS3DH_MISO 12
#define LIS3DH_MOSI 11
#define LIS3DH_CS 10

// Pinul la care sunt conectate LED-urile
#define PIN 8

// Numarul total de LED-uri
#define NUMPIXELS 5

// Imaginea inserata
#define IMAGE_A "sageata5.bmp"

// Definirea culorilor
#define BACKGROUND_COLOR __builtin_bswap16(ARCADA_BLACK)
#define BORDER_COLOR __builtin_bswap16(ARCADA_BLUE)
#define PLOT_COLOR_1 __builtin_bswap16(ARCADA_YELLOW)
#define TITLE_COLOR __builtin_bswap16(ARCADA_WHITE)
#define TICKTEXT_COLOR __builtin_bswap16(ARCADA_WHITE)
#define TICKLINE_COLOR __builtin_bswap16(ARCADA_DARKGREY)


// Definirea ariei de plotare
#define PLOT_TOPBUFFER 20
#define PLOT_LEFTBUFFER 40
#define PLOT_BOTTOMBUFFER 20
#define PLOT_W (ARCADA_TFT_WIDTH - PLOT_LEFTBUFFER)
#define PLOT_H (ARCADA_TFT_HEIGHT - PLOT_BOTTOMBUFFER - PLOT_TOPBUFFER)

// Pinul folosit pentru plotare
#define ANALOG_INPUT A4

// Intervalul de intarzierea in milisecunde
#define DELAY_PER_SAMPLE 50

// Pragul superior pentru detectarea bătăilor
#define UpperThreshold 520

// Pragul inferior pentru detectarea bătăilor
#define LowerThreshold 516

// Numarul de componente ale meniului
#define NUM_SELECTIONS 4


CircularBuffer<float, PLOT_W> data_buffer;  // Buffer pentru datele grafice

bool BPMTiming = false;     // Indicator pentru măsurarea BPM
bool BeatComplete = false;  // Indicator pentru terminarea unei bătăi
int LastTime2 = 0;           // Timpul ultimei bătăi
int BPM = 0;                // Frecvența cardiacă în BPM
int value = 0;
int beat, f_beat, u_beat;
int i;
int average=0;
int ThisTime;
int s=0,cnt=0;//s=suma dintre bataile inregistrate //cnt=inregistreaza cate valori au fost 
Adafruit_NeoPixel pixels = Adafruit_NeoPixel(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);

Adafruit_LIS3DH lis = Adafruit_LIS3DH();

const char *selection[NUM_SELECTIONS] = { "Test culori", "Test orientare", "Masurare ritm cardiac", "Joc de cuvinte" };




// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 70 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {

 pixels.begin();
  pixels.setBrightness(15);
  pixels.show();
  while (!Serial) delay(10);
  arcada.filesysBeginMSD();
  Serial.begin(115200);
  Serial.println("Controls Test");
  Serial.println("Hello! Arcada Menu test");
  if (!arcada.arcadaBegin()) {
    Serial.print("Failed to begin");
    while (1)
      ;
  }

  arcada.displayBegin();
  Serial.println("Arcada display begin");
  arcada.setBacklight(255);

  if (!arcada.createFrameBuffer(ARCADA_TFT_WIDTH, ARCADA_TFT_HEIGHT)) {
    Serial.print("Failed to allocate framebuffer");
    while (1)
      ;
  }
  if (arcada.filesysBegin()) {
    Serial.println("Found filesystem!");
  } else {
    arcada.haltBox("No filesystem found! For QSPI flash, load CircuitPython. For SD cards, format with FAT");
  }
  if (!lis.begin(0x18)) {
    Serial.println("Couldnt start");
    while (1) yield();
  }
  Serial.println("LIS3DH found!");

 
  
 


  
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_tiny_conv_micro_features_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::ops::micro::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_FULLY_CONNECTED,
      tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                                       tflite::ops::micro::Register_SOFTMAX());

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize,
      error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != kFeatureSliceCount) ||
      (model_input->dims->data[2] != kFeatureSliceSize) ||
      (model_input->type != kTfLiteUInt8)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return;
  }

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 model_input->data.uint8);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer(error_reporter);
  recognizer = &static_recognizer;

  previous_time = 0;
}

uint32_t timestamp = 0;
void loop() {
   uint8_t justPressed = arcada.justPressedButtons();
  uint8_t justReleased = arcada.justReleasedButtons();
  (void)justPressed;
  (void)justReleased;
  uint8_t buttons = arcada.readButtons();

  Serial.print("Pressed: ");

  arcada.display->fillScreen(ARCADA_RED);
  arcada.display->setTextColor(ARCADA_BLACK);
  arcada.display->setCursor(7, 15);
  arcada.display->setTextSize(1.5);
  arcada.display->print("Alegeti varianata dorita:");

  pixels.setPixelColor(0, pixels.Color(255, 0, 0));    // LED roșu
  pixels.setPixelColor(1, pixels.Color(255, 255, 0));  // LED galben
  pixels.setPixelColor(2, pixels.Color(0, 255, 0));    // LED verde
  pixels.setPixelColor(3, pixels.Color(0, 0, 255));    // LED albastru
  pixels.setPixelColor(4, pixels.Color(148, 0, 211));  // LED violet
  pixels.show();

  uint8_t selected = arcada.menu(selection, NUM_SELECTIONS, ARCADA_WHITE, ARCADA_BLACK);

  int randomNumber;

  char message[80];
  sprintf(message, "Selected '%s'", selection[selected]);

  unsigned long previousMillis = 0;
  unsigned long startTime;
  unsigned long int_cal = 6000;
  unsigned long interval = 11000;
  if (selected == 0) {
    uint16_t v[] = { ARCADA_BLACK, ARCADA_GREEN, ARCADA_BLUE, ARCADA_NAVY, ARCADA_DARKGREEN, ARCADA_DARKCYAN, ARCADA_MAROON, ARCADA_PURPLE, ARCADA_OLIVE, ARCADA_LIGHTGREY, ARCADA_DARKGREY, ARCADA_CYAN, ARCADA_RED, ARCADA_MAGENTA, ARCADA_YELLOW, ARCADA_WHITE, ARCADA_ORANGE, ARCADA_GREENYELLOW, ARCADA_PINK };
    Serial.print("Pressed: ");

    arcada.display->fillScreen(v[random(18)]);
    randomNumber = random(10);
    arcada.display->setTextColor(v[random(18)]);
    arcada.display->setCursor(64, 5);
    arcada.display->setTextSize(6);
    arcada.display->print(randomNumber);
    char mes[30];
    int aux = random(10);
    sprintf(mes, "Numarul este '%d' ?\n ADEVARAT (stanga)\n FALS (dreapta)", aux);
    arcada.infoBox(mes);

    uint8_t currentIndex = 0;
    uint8_t corect = 0;
    uint8_t gresit = 0;

    while (!(buttons & ARCADA_BUTTONMASK_LEFT & ARCADA_BUTTONMASK_RIGHT)) {
      if (buttons & ARCADA_BUTTONMASK_LEFT) {
        if (aux == randomNumber) {
          arcada.display->fillScreen(ARCADA_GREEN);
          arcada.display->setTextColor(ARCADA_BLACK);
          arcada.display->setCursor(40, 55);
          arcada.display->setTextSize(1);
          arcada.display->print("Raspuns Corect");
          for (int i = 0; i < NUMPIXELS; i++) {
            pixels.setPixelColor(i, pixels.Color(0, 255, 0));
          }
          pixels.show();
          delay(3000);
          for (int i = 0; i < NUMPIXELS; i++) {
            pixels.setPixelColor(i, pixels.Color(0, 0, 0));
          }
          pixels.show();

          corect++;
          previousMillis = millis();
          currentIndex++;
          if (currentIndex >= 5) {
            previousMillis = millis();
            if(corect==1)
            sprintf(mes, "Aveti un singur raspuns corect si %d raspunsuri gresite.", gresit);
            else if(gresit==1)
            sprintf(mes, "Aveti %d raspunsuri corecte si un singur raspuns gresit.", corect);
            else if(corect==0)
            sprintf(mes, "Nu aveti niciun raspuns corect.");
            else if(gresit==0)
            sprintf(mes, "Toate raspunsurile sunt corecte.");
            else sprintf(mes, "Aveti %d raspunsuri corecte si %d raspunsuri gresite.", corect, gresit);
            arcada.infoBox(mes);
            break;
          }
          arcada.display->fillScreen(v[random(18)]);
          randomNumber = random(10);
          arcada.display->setTextColor(v[random(18)]);
          arcada.display->setCursor(64, 5);
          arcada.display->setTextSize(6);
          arcada.display->print(randomNumber);
          aux = random(10);
          sprintf(mes, "Numarul este '%d' ?\n ADEVARAT (stanga)\n FALS (dreapta)", aux);
          arcada.infoBox(mes);
        } else {
          arcada.display->fillScreen(ARCADA_RED);
          arcada.display->setTextColor(ARCADA_BLACK);
          arcada.display->setCursor(40, 55);
          arcada.display->setTextSize(1);
          arcada.display->print("Raspuns Gresit");
          for (int i = 0; i < NUMPIXELS; i++) {
            pixels.setPixelColor(i, pixels.Color(255, 0, 0));
          }
          pixels.show();
          delay(3000);
          for (int i = 0; i < NUMPIXELS; i++) {
            pixels.setPixelColor(i, pixels.Color(0, 0, 0));
          }
          pixels.show();
          gresit++;
          previousMillis = millis();
          currentIndex++;
          if (currentIndex >= 5) {
            previousMillis = millis();
   //         sprintf(mes, "Aveti %d raspunsuri corecte si %d raspunsuri gresite.", corect, gresit);
   if(corect==1)
            sprintf(mes, "Aveti un singur raspuns corect si %d raspunsuri gresite.", gresit);
            else if(gresit==1)
            sprintf(mes, "Aveti %d raspunsuri corecte si un singur raspuns gresit.", corect);
            else if(corect==0)
            sprintf(mes, "Nu aveti niciun raspuns corect.");
            else if(gresit==0)
            sprintf(mes, "Toate raspunsurile sunt corecte.");
            else sprintf(mes, "Aveti %d raspunsuri corecte si %d raspunsuri gresite.", corect, gresit);
            arcada.infoBox(mes);
            break;
          }
          arcada.display->fillScreen(v[random(18)]);
          randomNumber = random(10);
          arcada.display->setTextColor(v[random(18)]);
          arcada.display->setCursor(64, 5);
          arcada.display->setTextSize(6);
          arcada.display->print(randomNumber);
          aux = random(10);
          sprintf(mes, "Numarul este '%d' ?\n ADEVARAT (stanga)\n FALS (dreapta)", aux);
          arcada.infoBox(mes);
        }
      } else if (buttons & ARCADA_BUTTONMASK_RIGHT) {
        if (aux == randomNumber) {
          arcada.display->fillScreen(ARCADA_RED);
          arcada.display->setTextColor(ARCADA_BLACK);
          arcada.display->setCursor(40, 55);
          arcada.display->setTextSize(1);
          arcada.display->print("Raspuns Gresit");
          for (int i = 0; i < NUMPIXELS; i++) {
            pixels.setPixelColor(i, pixels.Color(255, 0, 0));
          }
          pixels.show();
          delay(3000);
          for (int i = 0; i < NUMPIXELS; i++) {
            pixels.setPixelColor(i, pixels.Color(0, 0, 0));
          }
          pixels.show();
          gresit++;
          previousMillis = millis();
          currentIndex++;
          delay(3000);
          if (currentIndex >= 5) {
            previousMillis = millis();
   //         sprintf(mes, "Aveti %d raspunsuri corecte si %d raspunsuri gresite.", corect, gresit);
            if(corect==1)
            sprintf(mes, "Aveti un singur raspuns corect si %d raspunsuri gresite.", gresit);
            else if(gresit==1)
            sprintf(mes, "Aveti %d raspunsuri corecte si un singur raspuns gresit.", corect);
            else if(corect==0)
            sprintf(mes, "Nu aveti niciun raspuns corect.");
            else if(gresit==0)
            sprintf(mes, "Toate raspunsurile sunt corecte.");
            else sprintf(mes, "Aveti %d raspunsuri corecte si %d raspunsuri gresite.", corect, gresit);
            arcada.infoBox(mes);
            break;
          }
          arcada.display->fillScreen(v[random(18)]);
          randomNumber = random(10);
          arcada.display->setTextColor(v[random(18)]);
          arcada.display->setCursor(64, 5);
          arcada.display->setTextSize(6);
          arcada.display->print(randomNumber);
          aux = random(10);
          sprintf(mes, "Numarul este '%d' ?\n ADEVARAT (stanga)\n FALS (dreapta)", aux);
          arcada.infoBox(mes);
        } else {
          arcada.display->fillScreen(ARCADA_GREEN);
          arcada.display->setTextColor(ARCADA_BLACK);
          arcada.display->setCursor(40, 55);
          arcada.display->setTextSize(1);
          arcada.display->print("Raspuns Corect");
          for (int i = 0; i < NUMPIXELS; i++) {
            pixels.setPixelColor(i, pixels.Color(0, 255, 0));
          }
          pixels.show();
          delay(3000);
          for (int i = 0; i < NUMPIXELS; i++) {
            pixels.setPixelColor(i, pixels.Color(0, 0, 0));
          }
          pixels.show();
          corect++;
          previousMillis = millis();
          currentIndex++;
          if (currentIndex >= 5) {
            previousMillis = millis();
    //        sprintf(mes, "Aveti %d raspunsuri corecte si %d raspunsuri gresite.", corect, gresit);
            if(corect==1)
            sprintf(mes, "Aveti un singur raspuns corect si %d raspunsuri gresite.", gresit);
            else if(gresit==1)
            sprintf(mes, "Aveti %d raspunsuri corecte si un singur raspuns gresit.", corect);
            else if(corect==0)
            sprintf(mes, "Nu aveti niciun raspuns corect.");
            else if(gresit==0)
            sprintf(mes, "Toate raspunsurile sunt corecte.");
            else sprintf(mes, "Aveti %d raspunsuri corecte si %d raspunsuri gresite.", corect, gresit);
            arcada.infoBox(mes);
            break;
          }
          arcada.display->fillScreen(v[random(18)]);
          randomNumber = random(10);
          arcada.display->setTextColor(v[random(18)]);
          arcada.display->setCursor(64, 5);
          arcada.display->setTextSize(6);
          arcada.display->print(randomNumber);
          aux = random(10);
          sprintf(mes, "Numarul este '%d' ?\n ADEVARAT (stanga)\n FALS (dreapta)", aux);
          arcada.infoBox(mes);
        }
      }

      unsigned long currentMillis = millis();
      buttons = arcada.readButtons();
      if (currentMillis - previousMillis >= interval) {
        previousMillis = millis();
      }
    }
  }
  if (selected == 1) {
    char mes1[30];

    const char *v2[] = { "fata", "spate", "stanga", "dreapta", "sus", "jos" };

    int length = sizeof(v2) / sizeof(v2[0]);  // Obține numărul de stringuri în șirul de stringuri
    const char *randomString;
    uint8_t currentIndex2 = 0;
    uint8_t corect2 = 0;
    uint8_t gresit2 = 0;
    while (currentIndex2 < 5) {
      currentIndex2++;

      randomString = v2[random(length)];  // Obține stringul aleatoriu

      Serial.println(randomString);  // Afișează stringul aleatoriu în monitorul serial

      sensors_event_t event;
      //int ok=1;
      lis.getEvent(&event);
      if (!(buttons & ARCADA_BUTTONMASK_LEFT)) {
        const char *imagefile = IMAGE_A;
        arcada.display->fillScreen(ARCADA_WHITE);
        ImageReturnCode stat = arcada.drawBMP((char *)imagefile, 0, 0);
        Serial.println();
        sprintf(mes1, "Orientati sageata in %s ", randomString);
        arcada.infoBox(mes1);
        lis.read();
        lis.getEvent(&event);




        float x = event.acceleration.x / 9.8;
        float y = event.acceleration.y / 9.8;
        float z = event.acceleration.z / 9.8;

        Serial.print("\t\tX: ");
        Serial.print(x);
        Serial.print(" \tY: ");
        Serial.print(y);
        Serial.print(" \tZ: ");
        Serial.print(z);


        if (strcmp(randomString, "jos") == 0) {
          if (y <= -0.85) {
            Serial.print("corect");
            arcada.display->fillScreen(ARCADA_GREEN);
            arcada.display->setTextColor(ARCADA_BLACK);
            arcada.display->setCursor(40, 55);
            arcada.display->setTextSize(1);
            arcada.display->print("Raspuns Corect");
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 255, 0));
            }
            pixels.show();
            corect2++;
            delay(3000);
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 0, 0));
            }
            pixels.show();
          } else {
            Serial.print("gresit");
            arcada.display->fillScreen(ARCADA_RED);
            arcada.display->setTextColor(ARCADA_BLACK);
            arcada.display->setCursor(40, 55);
            arcada.display->setTextSize(1);
            arcada.display->print("Raspuns Gresit");
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(255, 0, 0));
            }
            pixels.show();
            delay(3000);
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 0, 0));
            }
            pixels.show();
            gresit2++;
          }
        }
        if (strcmp(randomString, "sus") == 0) {
          if (y >= 0.85) {
            Serial.print("corect");
            arcada.display->fillScreen(ARCADA_GREEN);
            arcada.display->setTextColor(ARCADA_BLACK);
            arcada.display->setCursor(40, 55);
            arcada.display->setTextSize(1);
            arcada.display->print("Raspuns Corect");
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 255, 0));
            }
            pixels.show();
            corect2++;
            delay(3000);
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 0, 0));
            }
            pixels.show();
          } else {
            Serial.print("gresit");
            arcada.display->fillScreen(ARCADA_RED);
            arcada.display->setTextColor(ARCADA_BLACK);
            arcada.display->setCursor(40, 55);
            arcada.display->setTextSize(1);
            arcada.display->print("Raspuns Gresit");
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(255, 0, 0));
            }
            pixels.show();
            delay(3000);
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 0, 0));
            }
            pixels.show();
            gresit2++;
          }
        }
        if (strcmp(randomString, "fata") == 0) {
          if (z <= -0.85) {
            Serial.print("corect");
            arcada.display->fillScreen(ARCADA_GREEN);
            arcada.display->setTextColor(ARCADA_BLACK);
            arcada.display->setCursor(40, 55);
            arcada.display->setTextSize(1);
            arcada.display->print("Raspuns Corect");
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 255, 0));
            }
            pixels.show();
            corect2++;
            delay(3000);
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 0, 0));
            }
            pixels.show();
          } else {
            Serial.print("gresit");
            arcada.display->fillScreen(ARCADA_RED);
            arcada.display->setTextColor(ARCADA_BLACK);
            arcada.display->setCursor(40, 55);
            arcada.display->setTextSize(1);
            arcada.display->print("Raspuns Gresit");
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(255, 0, 0));
            }
            pixels.show();
            delay(3000);
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 0, 0));
            }
            pixels.show();
            gresit2++;
          }
        }
        if (strcmp(randomString, "spate") == 0) {
          if (z >= 0.85) {
            Serial.print("corect");
            arcada.display->fillScreen(ARCADA_GREEN);
            arcada.display->setTextColor(ARCADA_BLACK);
            arcada.display->setCursor(40, 55);
            arcada.display->setTextSize(1);
            arcada.display->print("Raspuns Corect");
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 255, 0));
            }
            pixels.show();
            corect2++;
            delay(3000);
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 0, 0));
            }
            pixels.show();
          } else {
            Serial.print("gresit");
            arcada.display->fillScreen(ARCADA_RED);
            arcada.display->setTextColor(ARCADA_BLACK);
            arcada.display->setCursor(40, 55);
            arcada.display->setTextSize(1);
            arcada.display->print("Raspuns Gresit");
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(255, 0, 0));
            }
            pixels.show();
            delay(3000);
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 0, 0));
            }
            pixels.show();
            gresit2++;
          }
        }
        if (strcmp(randomString, "stanga") == 0) {
          if (x <= -0.85) {
            Serial.print("corect");
            arcada.display->fillScreen(ARCADA_GREEN);
            arcada.display->setTextColor(ARCADA_BLACK);
            arcada.display->setCursor(40, 55);
            arcada.display->setTextSize(1);
            arcada.display->print("Raspuns Corect");
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 255, 0));
            }
            pixels.show();
            corect2++;
            delay(3000);
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 0, 0));
            }
            pixels.show();
          } else {
            Serial.print("gresit");
            arcada.display->fillScreen(ARCADA_RED);
            arcada.display->setTextColor(ARCADA_BLACK);
            arcada.display->setCursor(40, 55);
            arcada.display->setTextSize(1);
            arcada.display->print("Raspuns Gresit");
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(255, 0, 0));
            }
            pixels.show();
            delay(3000);
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 0, 0));
            }
            pixels.show();
            gresit2++;
          }
        }
        if (strcmp(randomString, "dreapta") == 0) {
          if (x >= 0.85) {
            Serial.print("corect");
            arcada.display->fillScreen(ARCADA_GREEN);
            arcada.display->setTextColor(ARCADA_BLACK);
            arcada.display->setCursor(40, 55);
            arcada.display->setTextSize(1);
            arcada.display->print("Raspuns Corect");
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 255, 0));
            }
            pixels.show();
            corect2++;
            delay(3000);
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 0, 0));
            }
            pixels.show();
          } else {
            Serial.print("gresit");
            arcada.display->fillScreen(ARCADA_RED);
            arcada.display->setTextColor(ARCADA_BLACK);
            arcada.display->setCursor(40, 55);
            arcada.display->setTextSize(1);
            arcada.display->print("Raspuns Gresit");
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(255, 0, 0));
            }
            pixels.show();
            delay(3000);
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 0, 0));
            }
            pixels.show();
            gresit2++;
          }
        }

    
      }
    }
    if (currentIndex2 == 5) {
      previousMillis = millis();
      char mes2[30];
       if(corect2==1)
            sprintf(mes2, "Aveti un singur raspuns corect si %d raspunsuri gresite.", gresit2);
            else if(gresit2==1)
            sprintf(mes2, "Aveti %d raspunsuri corecte si un singur raspuns gresit.", corect2);
            else if(corect2==0)
            sprintf(mes2, "Nu aveti niciun raspuns corect.");
            else if(gresit2==0)
            sprintf(mes2, "Toate raspunsurile sunt corecte.");
            else sprintf(mes2, "Aveti %d raspunsuri corecte si %d raspunsuri gresite.", corect2, gresit2);
      arcada.infoBox(mes2);
    }
  }
  if (selected == 2) {
    if (millis() - timestamp < DELAY_PER_SAMPLE) {
      return;
    }
    arcada.display->fillScreen(ARCADA_OLIVE);
    char mes2[30];
    arcada.display->fillScreen(ARCADA_OLIVE);

    sprintf(mes2, "Pozitionati degetul aratator pe senzorul de puls ");
    arcada.infoBox(mes2);

    startTime = millis();
    timestamp = millis();
    int OK = 0;
    int i1 = 6;
    int i2 = 11;

    unsigned long calStartTime = millis();
    unsigned long calInterval = 1000;

    unsigned long updateStartTime = millis();
    unsigned long updateInterval = 1000;

    

    while (1) {
      ThisTime=millis();
      value = analogRead(A4);
      //Serial.println(value);
      delay(60);
      if (value > UpperThreshold) {Serial.println(BPM);
      
        if (BeatComplete) {
          BPM=ThisTime-LastTime2;
          BPM=int(60/(float(BPM)/1000));
          BPMTiming = false;
          BeatComplete = false;
        }
        if (BPMTiming == false) {
          LastTime2 = millis();
          BPMTiming = true;
        }
      }
      if ((value < LowerThreshold) & (BPMTiming))  // Verificare dacă valoarea scade sub pragul inferior și BPM-ul se măsoară
        BeatComplete = true;                       // Indică că o bătaie s-a încheiat

      // Afișare a frecvenței cardiace în Serial Monitor
     // Serial.print("Frecventa cardiaca: ");
     // Serial.print(beat);
      //Serial.println(" BPM");
    //  Serial.println(BPM);
   
      
      


      unsigned long currentTime2 = millis();
      if (OK == 0) {
        if (currentTime2 - startTime >= int_cal) {
          f_beat = BPM;
          
          
          OK = 1;
          startTime = currentTime2;
        } else {
          arcada.display->fillScreen(ARCADA_YELLOW);
          arcada.display->setTextColor(ARCADA_BLACK);
          arcada.display->setCursor(30, 30);
          arcada.display->setTextSize(1.5);
          arcada.display->print("Se calibreaza in: ");
          arcada.display->setCursor(70, 60);
          arcada.display->setTextSize(4);
          arcada.display->print(i1);
          if (millis() - calStartTime >= calInterval) {
            i1--;
            calStartTime = millis();
          }
        }
      } else {
        if (currentTime2 - startTime >= interval) {
          //u_beat = BPM - f_beat;
       /*for(int m=0;m<100;m++){
        average=average+ BPM;
      }
      average=average/100;*/
         
          
          startTime = currentTime2;
          break;
        } else { 
          data_buffer.push(value);
           if(BPM<150 && BPM>50)
          {s=s+BPM;
          cnt++;}
           u_beat=s/cnt;//media aritmetica intre suna si numarul valorilor inregistrate 
          String text = "Inca: " + String(i2) + "(s)";
          const char *text_cstr = text.c_str();
          plotBuffer(arcada.getCanvas(), data_buffer, text_cstr);
          arcada.blitFrameBuffer(0, 0, false, true);
          if (millis() - updateStartTime >= updateInterval) {
            i2--;
            updateStartTime = millis();
          }
        }
      }
    }
    arcada.display->fillScreen(ARCADA_PINK);
    arcada.display->setTextColor(ARCADA_BLACK);
    arcada.display->setCursor(20, 30);
    arcada.display->setTextSize(1.5);
    arcada.display->print("Pulsul dumneavoastra: ");
    arcada.display->setCursor(70, 60);
    arcada.display->setTextSize(4);
   
    arcada.display->print(u_beat);
   // Serial.println(BPM);
    delay(5000);
    int k=0;
    if(u_beat>110)
    {arcada.display->fillScreen(ARCADA_RED);
    arcada.display->setTextColor(ARCADA_BLACK);
    arcada.display->setCursor(15, 55);
    arcada.display->setTextSize(1.75);
    arcada.display->print("Alerta de puls ridicat");
    k=0;
    while(k!=6)
    {for (int i = 0; i < NUMPIXELS; i++) {
            pixels.setPixelColor(i, pixels.Color(255, 0, 0));
          }
          pixels.show();
          delay(500);
          for (int i = 0; i < NUMPIXELS; i++) {
            pixels.setPixelColor(i, pixels.Color(0, 0, 0));
          }
          pixels.show();
          delay(500);
          k++;}
    //delay(5000);
    }
    if(u_beat<55)
    {arcada.display->fillScreen(ARCADA_RED);
    arcada.display->setTextColor(ARCADA_BLACK);
    arcada.display->setCursor(15, 55);
    arcada.display->setTextSize(1.75);
    arcada.display->print("Alerta de puls scazut");
    k=0;
    while(k!=6)
    {for (int i = 0; i < NUMPIXELS; i++) {
            pixels.setPixelColor(i, pixels.Color(255, 0, 0));
          }
          pixels.show();
          delay(500);
          for (int i = 0; i < NUMPIXELS; i++) {
            pixels.setPixelColor(i, pixels.Color(0, 0, 0));
          }
          pixels.show();
          delay(500);
          k++;}
    //delay(5000);
    }
  u_beat=0;
  s=0;
  cnt=0;}
  if (selected == 3) {
    char mes[30];
     const char* v[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
   int length = sizeof(v) / sizeof(v[0]);  // Obține numărul de stringuri în șirul de stringuri
const char *randomString;
randomString = v[random(length)];

arcada.display->fillScreen(ARCADA_OLIVE);
arcada.display->setCursor(2, 50);
arcada.display->setTextSize(1);
arcada.display->setTextColor(ARCADA_BLACK);
arcada.display->print("Pronuntati corect cuvantul ");
arcada.display->print(randomString);
arcada.display->println(" in limba engleza");


   
    while (1) { 
  // Fetch the spectrogram for the current time.
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      error_reporter, previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    error_reporter->Report("Feature generation failed");
    return;
  }
  previous_time = current_time;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  if (how_many_new_slices == 0) {
    return;
  }

  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Obtain a pointer to the output tensor
  TfLiteTensor* output = interpreter->output(0);
  // Determine whether a command was recognized based on the output of inference
  const char* found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;
  TfLiteStatus process_status = recognizer->ProcessLatestResults(
      output, current_time, &found_command, &score, &is_new_command);
  if (process_status != kTfLiteOk) {
    error_reporter->Report("RecognizeCommands::ProcessLatestResults() failed");
    return;
  }
  // Do something based on the recognized command. The default implementation
  // just prints to the error console, but you should replace this with your
  // own function for a real application.
if (is_new_command) {
    error_reporter->Report("Heard %s (%d) @%dms", found_command, score,
                           current_time);
                           sprintf(mes, "Apasati butonul 'a' pentru a reveni la meniul principal ");
        

        if(found_command==randomString){
          Serial.print("Ati pronuntat corect !");
            arcada.display->fillScreen(ARCADA_GREEN);
            arcada.display->setTextColor(ARCADA_BLACK);
            arcada.display->setCursor(5, 50);
            arcada.display->setTextSize(1);
            arcada.display->print("Ati pronuntat corect !");
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 255, 0));
            }
            pixels.show();
            delay(3000);
       break; 
       }
     
      else {
            Serial.print("Nu ati pronuntat corespunzator, mai incercati !");
            arcada.display->fillScreen(ARCADA_RED);
            arcada.display->setTextColor(ARCADA_BLACK);
            arcada.display->setCursor(5, 50);
            arcada.display->setTextSize(1);
            arcada.display->print("Nu ati pronuntat optim !");
             arcada.display->setCursor(5, 60);
            arcada.display->println( "Mai incercati !");
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(255, 0, 0));
            }
            pixels.show();
            error_reporter->Report("Heard %s (%d) @%dms", found_command, score,
                           current_time);
            }
        
       //arcada.infoBox(mes);
   }
/*randomString = v[random(length)];
   arcada.display->fillScreen(ARCADA_OLIVE);
arcada.display->setCursor(2, 50);
arcada.display->setTextSize(1);
arcada.display->setTextColor(ARCADA_BLACK);
arcada.display->print("Pronuntati corect cuvantul ");
arcada.display->print(randomString);
arcada.display->println(" in limba engleza");
error_reporter->Report("Heard %s (%d) @%dms", found_command, score,
                           current_time);

if (is_new_command) {
    error_reporter->Report("Heard %s (%d) @%dms", found_command, score,
                           current_time);
                           sprintf(mes, "Apasati butonul 'a' pentru a reveni la meniul principal ");
        

        if(found_command==randomString){
          Serial.print("Ati pronuntat corect !");
            arcada.display->fillScreen(ARCADA_GREEN);
            arcada.display->setTextColor(ARCADA_BLACK);
            arcada.display->setCursor(5, 50);
            arcada.display->setTextSize(1);
            arcada.display->print("Ati pronuntat corect !");
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(0, 255, 0));
            }
            pixels.show();
            delay(3000);
       break; }
     
      else {
            Serial.print("Nu ati pronuntat corespunzator, mai incercati !");
            arcada.display->fillScreen(ARCADA_RED);
            arcada.display->setTextColor(ARCADA_BLACK);
            arcada.display->setCursor(5, 50);
            arcada.display->setTextSize(1);
            arcada.display->print("Nu ati pronuntat optim !");
             arcada.display->setCursor(5, 60);
            arcada.display->println( "Mai incercati !");
            for (int i = 0; i < NUMPIXELS; i++) {
              pixels.setPixelColor(i, pixels.Color(255, 0, 0));
            }
            pixels.show();
            error_reporter->Report("Heard %s (%d) @%dms", found_command, score,
                           current_time);
            }
        
       //arcada.infoBox(mes);
   }*/


   }
   
   /* sprintf(mes, "Numarul este '' ?\n ADEVARAT (stanga)\n FALS (dreapta)");
    arcada.infoBox(mes);
while (!(buttons & ARCADA_BUTTONMASK_LEFT & ARCADA_BUTTONMASK_RIGHT)) {
      if (buttons & ARCADA_BUTTONMASK_LEFT) {
    const char* v[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
    const char* randomnumber = nullptr;

    arcada.display->fillScreen(ARCADA_OLIVE);
    arcada.display->setCursor(50, 50);
    arcada.display->setTextSize(3);
    randomnumber = v[random(10)];
    arcada.display->print(randomnumber);
RespondToCommand(error_reporter, current_time, found_command, score, is_new_command);
   char cuvant_rostit[2];  // Șirul pentru cuvântul rostit și terminatorul '\0'

cuvant_rostit[0] = found_command[0];

    // Compararea cuvântului rostit cu cuvântul corect
    if (cuvant_rostit && strcmp(cuvant_rostit, randomnumber) == 0) {
        // Afișați "Raspuns Corect" doar în cazul răspunsului corect
        arcada.display->fillScreen(ARCADA_OLIVE);
        arcada.display->setCursor(50, 50);
        arcada.display->setTextSize(3);
        arcada.display->print("Raspuns Corect");

        // Continuați să procesați comenzi și recunoașteți vocea în timp real
        RespondToCommand(error_reporter, current_time, found_command, score, is_new_command);
    } else {
        // Afișați "Raspuns Incorect" sau orice alt mesaj pentru răspunsul greșit
        arcada.display->fillScreen(ARCADA_OLIVE);
        arcada.display->setCursor(50, 50);
        arcada.display->setTextSize(3);
        arcada.display->print("Raspuns Incorect");

        // Puteți aștepta o perioadă scurtă sau faceți alte acțiuni în cazul răspunsului greșit
        delay(10000);  // Așteaptă 2 secunde
    }*/
}
               

}

void plotBuffer(GFXcanvas16 *_canvas, CircularBuffer<float, PLOT_W> &buffer, const char *title) {
  _canvas->fillScreen(BACKGROUND_COLOR);
  _canvas->drawLine(PLOT_LEFTBUFFER - 1, PLOT_TOPBUFFER,
                    PLOT_LEFTBUFFER - 1, PLOT_H + PLOT_TOPBUFFER, BORDER_COLOR);
  _canvas->drawLine(PLOT_LEFTBUFFER - 1, PLOT_TOPBUFFER + PLOT_H + 1,
                    ARCADA_TFT_WIDTH, PLOT_TOPBUFFER + PLOT_H + 1, BORDER_COLOR);
  _canvas->setTextSize(2);
  _canvas->setTextColor(TITLE_COLOR);
  uint16_t title_len = strlen(title) * 12;
  _canvas->setCursor((_canvas->width() - title_len) / 2, 0);
  _canvas->print(title);

  float minY = 0;
  float maxY = 0;

  if (buffer.size() > 0) {
    maxY = minY = buffer[0];
  }
  for (int i = 0; i < buffer.size(); i++) {
    minY = min(minY, buffer[i]);
    maxY = max(maxY, buffer[i]);
  }

  float MIN_DELTA = 10.0;
  if (maxY - minY < MIN_DELTA) {
    float mid = (maxY + minY) / 2;
    maxY = mid + MIN_DELTA / 2;
    minY = mid - MIN_DELTA / 2;
  } else {
    float extra = (maxY - minY) / 10;
    maxY += extra;
    minY -= extra;
  }

  printTicks(_canvas, 5, minY, maxY);

  int16_t last_y = 0, last_x = 0;
  for (int i = 0; i < buffer.size(); i++) {
    int16_t y = map(buffer[i], minY, maxY, PLOT_TOPBUFFER + PLOT_H, PLOT_TOPBUFFER);

    int16_t x = PLOT_LEFTBUFFER + i;
    if (i == 0) {
      last_y = y;
      last_x = x;
    }
    _canvas->drawLine(last_x, last_y, x, y, PLOT_COLOR_1);
    last_x = x;
    last_y = y;
  }
}


void printTicks(GFXcanvas16 *_canvas, uint8_t ticks, float minY, float maxY) {
  _canvas->setTextSize(1);
  _canvas->setTextColor(TICKTEXT_COLOR);
  for (int t = 0; t < ticks; t++) {
    float v = map(t, 0, ticks - 1, minY, maxY);
    uint16_t y = map(t, 0, ticks - 1, ARCADA_TFT_HEIGHT - PLOT_BOTTOMBUFFER - 4, PLOT_TOPBUFFER);
    printLabel(_canvas, 0, y, v);
    uint16_t line_y = map(t, 0, ticks - 1, ARCADA_TFT_HEIGHT - PLOT_BOTTOMBUFFER, PLOT_TOPBUFFER);
    _canvas->drawLine(PLOT_LEFTBUFFER, line_y, ARCADA_TFT_WIDTH, line_y, TICKLINE_COLOR);
  }
}

void printLabel(GFXcanvas16 *_canvas, uint16_t x, uint16_t y, float val) {
  (void)x;

  char label[20];
  if (abs(val) < 1) {
    snprintf(label, 19, "%0.2f", val);
  } else if (abs(val) < 10) {
    snprintf(label, 19, "%0.1f", val);
  } else {
    snprintf(label, 19, "%d", (int)val);
  }

  _canvas->setCursor(PLOT_LEFTBUFFER - strlen(label) * 6 - 5, y);
  _canvas->print(label);
}
