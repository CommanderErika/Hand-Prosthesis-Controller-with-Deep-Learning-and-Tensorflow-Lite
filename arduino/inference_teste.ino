#include "model.h"
#include "utils.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

//#include <TimeLib.h>

// General data //
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::MicroErrorReporter error;
constexpr int tensor_arena_size = 60 * 1024; // aumentar isso aqui
static uint8_t tensor_arena[tensor_arena_size];
//uint8_t *tensor_arena = nullptr;
//byte tensor_arena[tensor_arena_size] __attribute__ ((aligned(16)));

// Input and output informations //
TfLiteTensor* input_tensor_1 = nullptr;
TfLiteTensor* input_tensor_2 = nullptr;
TfLiteTensor* output_tensor = nullptr;
float i_scale_1 = 0.15294118;
int32_t i_zero_point_1 = -128;
float i_scale_2 = 0.1764706;
int32_t i_zero_point_2 = -128;
int input = 0;
float min_;
size_t idx_selected;

void setup() {

  // Serial port //
  //_ontouch1200bps_();
  Serial.begin(9600);
  Serial.println("Teste");
  
  // Allocating Tensor //
  //tensor_arena = (uint8_t *)malloc(tensor_arena_size);
  
  // Importing Model //
  model = tflite::GetModel(model_quant_tflite);
  
  // Resolver //
  tflite::AllOpsResolver ops_resolver;
  
  // Interpreter //
  interpreter = new tflite::MicroInterpreter(model, ops_resolver, tensor_arena,
      tensor_arena_size, &error);
  
  interpreter->AllocateTensors();
  
  input_tensor_1 = interpreter->input(0);
  input_tensor_2 = interpreter->input(1);
  output_tensor = interpreter->output(0);
  
  // Input and output informations //
  
  // Quantization //
  /*
  const auto* i_quantization_1 = reinterpret_cast<TfLiteAffineQuantization*>(input_tensor_1->quantization.params);
  const auto* i_quantization_2 = reinterpret_cast<TfLiteAffineQuantization*>(input_tensor_2->quantization.params);
  i_scale_1 = i_quantization_1->scale->data[0];
  i_zero_point_1 = i_quantization_1->zero_point->data[0];
  i_scale_2 = i_quantization_2->scale->data[0];
  i_zero_point_2 = i_quantization_2->zero_point->data[0];
  */
  
}

void loop() {

  // Setting inputs //
  Serial.println(" ================================ ");
  Serial.println(" ======== Pegando entrada ======= ");
  //Serial.println("Aplicando MRMS...");
  //Moving_rms(50, signal_);
  
  for (int i = 0; i < 128 * 20; i++){
    input_tensor_1->data.int8[i] = (int8_t)((input_1[i]/i_scale_1) + i_zero_point_1);
    input_tensor_2->data.int8[i] = (int8_t)((input_2[i]/i_scale_2) + i_zero_point_2);
  }
  // Invoking Model to inference //
  Serial.println(" ====== Fazendo inferência ====== ");
  float time_ = (float)(millis());
  TfLiteStatus invoke_status = interpreter->Invoke();
  Serial.print("Tempo de inferência (ms): ");
  Serial.println(millis() - time_);

  // Getting Output //
  Serial.println(" =========== Resultado ========== ");
  min_ = output_tensor->data.int8[0];
  idx_selected = 0;
  for (int i = 0; i < 5; i++) {
    if(min_ < output_tensor->data.int8[i]){
      min_ = output_tensor->data.int8[i];
      idx_selected = i;
    }
    Serial.println(output_tensor->data.int8[i]);
  }

  // Decision //
  if(idx_selected == 0){
    Serial.println("Classe Cly");
  }else if(idx_selected == 1){
    Serial.println("Classe Hook");
  }else if(idx_selected == 2){
    Serial.println("Classe tip");
  }else if(idx_selected == 6){
    Serial.println("Classe palm");
  }else if(idx_selected == 3){
    Serial.println("Classe spher");
  }else if(idx_selected == 4){
    Serial.println("Classe lat");
  }

}
