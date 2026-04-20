#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum DenkflowResult {
  /**
   * Operation completed successfully.
   */
  DenkflowResult_Ok = 0,
  /**
   * A null pointer was provided where it was not expected.
   */
  DenkflowResult_NullPointer = -1,
  /**
   * A null pointer was expected but the pointer was already pointing to some data.
   */
  DenkflowResult_NonNullPointer = -2,
  /**
   * The given pointer is already being used for another object. The other object needs to be freed first.
   */
  DenkflowResult_PointerInUse = -3,
  /**
   * The given object does not exist or might have already been freed
   */
  DenkflowResult_ObjectDoesNotExist = -4,
  /**
   * The operation failed due to an invalid UTF-8 string.
   */
  DenkflowResult_InvalidUtf8 = -5,
  /**
   * An error occurred while handling the pipeline.
   */
  DenkflowResult_PipelineError = -6,
  /**
   * An error in onnxruntime occurred.
   */
  DenkflowResult_OnnxError = -7,
  /**
   * A provided image could not be parsed.
   */
  DenkflowResult_ImageError = -8,
  /**
   * An error in the receiver occurred.
   */
  DenkflowResult_ReceiverError = -9,
  /**
   * Error when converting a string into a C-String
   */
  DenkflowResult_NulError = -10,
  /**
   * Error parsing a UUID
   */
  DenkflowResult_UuidError = -11,
  /**
   * An object passed as a void pointer had the wrong type
   */
  DenkflowResult_ObjectHasWrongType = -12,
  /**
   * An error occurred while casting an array into a specific shape
   */
  DenkflowResult_ShapeError = -13,
  /**
   * An error occurred in Ort
   */
  DenkflowResult_OrtError = -14,
  /**
   * Error while parsing a string as an integer
   */
  DenkflowResult_ParseIntError = -15,
  /**
   * The operation failed for an unspecified reason. Check the last error to get more information.
   */
  DenkflowResult_Unspecified = -1000,
} DenkflowResult;

typedef enum EnumImagePatchesNodeResizeMode {
  EnumImagePatchesNodeResizeMode_CenterPadBlack = 0,
  EnumImagePatchesNodeResizeMode_Stretch = 1,
} EnumImagePatchesNodeResizeMode;

typedef enum EnumTensorType {
  EnumTensorType_Tensor = 0,
  EnumTensorType_ImageTensor = 1,
  EnumTensorType_BoundingBoxTensor = 2,
  EnumTensorType_OcrTensor = 3,
  EnumTensorType_ScalarTensor = 4,
  EnumTensorType_SegmentationMaskTensor = 5,
  EnumTensorType_InstanceSegmentationMaskTensor = 6,
} EnumTensorType;

typedef enum EnumTopicType {
  EnumTopicType_InternalConnection = 0,
  EnumTopicType_ExternalInput = 1,
  EnumTopicType_ExternalOuptut = 2,
} EnumTopicType;

typedef struct BaseTensor BaseTensor;

typedef struct BoundingBoxTensor BoundingBoxTensor;

typedef struct HubLicenseSource HubLicenseSource;

typedef struct ImageTensor ImageTensor;

typedef struct InitializedPipeline InitializedPipeline;

typedef struct InstanceSegmentationMaskTensor InstanceSegmentationMaskTensor;

typedef struct OcrTensor OcrTensor;

typedef struct OneTimeLicenseSource OneTimeLicenseSource;

typedef struct Pipeline Pipeline;

typedef struct ScalarTensor ScalarTensor;

typedef struct SegmentationMaskTensor SegmentationMaskTensor;

typedef struct TopicInformation {
  char *topic_name;
  enum EnumTopicType topic_type;
  enum EnumTensorType tensor_type;
} TopicInformation;

typedef struct TopicInformationArray {
  struct TopicInformation *topic_information;
  uintptr_t topic_information_length;
} TopicInformationArray;

typedef struct ConstTensorNodeParams {
  struct BaseTensor *tensor;
} ConstTensorNodeParams;

typedef struct InitializedConstTensorNodeParams {
  struct BaseTensor *tensor;
} InitializedConstTensorNodeParams;

typedef struct VirtualCameraNodeParams {
  const char *file_path;
} VirtualCameraNodeParams;

typedef struct InitializedVirtualCameraNodeParams {
  const char *file_path;
} InitializedVirtualCameraNodeParams;

typedef struct CStringWrapper {
  char *c_string;
} CStringWrapper;

typedef struct StringArray {
  struct CStringWrapper *strings;
  uintptr_t string_length;
} StringArray;

typedef struct BoundingBoxFilterNodeReference {
  char *name;
  char *output;
} BoundingBoxFilterNodeReference;

typedef struct ConstTensorNodeReference {
  char *name;
  char *output;
} ConstTensorNodeReference;

typedef struct ImageAnomalyDetectionNodeReference {
  char *name;
  char *anomaly_score_output;
  char *segmentation_output;
} ImageAnomalyDetectionNodeReference;

typedef struct ImageClassificationNodeReference {
  char *name;
  char *output;
} ImageClassificationNodeReference;

typedef struct ImageInstanceSegmentationNodeReference {
  char *name;
  char *bounding_box_output;
  char *segmentation_output;
} ImageInstanceSegmentationNodeReference;

typedef struct ImageObjectDetectionNodeReference {
  char *name;
  char *output;
} ImageObjectDetectionNodeReference;

typedef struct ImagePatchesNodeReference {
  char *name;
  char *output;
} ImagePatchesNodeReference;

typedef struct ImageResizeNodeReference {
  char *name;
  char *output;
} ImageResizeNodeReference;

typedef struct ImageSegmentationNodeReference {
  char *name;
  char *output;
} ImageSegmentationNodeReference;

typedef struct OCRNodeReference {
  char *name;
  char *output;
} OCRNodeReference;

typedef struct VirtualCameraNodeReference {
  char *name;
  char *output;
} VirtualCameraNodeReference;

typedef struct Receiver_Tensor Receiver_Tensor;

typedef struct ClassLabel {
  char *class_label_id;
  char *name;
  char *short_name;
  char *color;
} ClassLabel;

typedef struct BoundingBox {
  float x1;
  float y1;
  float x2;
  float y2;
  float angle_rad;
  float confidence;
  struct ClassLabel class_label;
  uintptr_t batch_index;
} BoundingBox;

typedef struct BoundingBoxResults {
  struct BoundingBox *bounding_boxes;
  uintptr_t bounding_boxes_length;
} BoundingBoxResults;

typedef struct Point {
  double x;
  double y;
} Point;

typedef struct Rect {
  struct Point top_left;
  struct Point bottom_right;
} Rect;

typedef struct SubContour {
  bool is_hole;
  struct Point *points;
  uintptr_t points_length;
} SubContour;

typedef struct Contour {
  struct SubContour *sub_contours;
  uintptr_t sub_contours_length;
} Contour;

typedef struct SegmentationObject {
  float confidence;
  struct Rect bounding_rect;
  struct Contour contour;
} SegmentationObject;

typedef struct SegmentationClass {
  struct ClassLabel class_label;
  struct SegmentationObject *segmentation_objects;
  uintptr_t segmentation_objects_length;
} SegmentationClass;

typedef struct SegmentationBatchElement {
  struct SegmentationClass *segmentation_classes;
  uintptr_t segmentation_classes_length;
} SegmentationBatchElement;

typedef struct SegmentationResults {
  struct SegmentationBatchElement *segmentation_batch_elements;
  uintptr_t segmentation_batch_elements_length;
} SegmentationResults;

typedef struct OcrResults {
  struct CStringWrapper *ocr_strings;
  uintptr_t ocr_strings_length;
} OcrResults;

typedef struct Scalar {
  float value;
  struct ClassLabel class_label;
} Scalar;

typedef struct ScalarBatchElement {
  struct Scalar *scalars;
  uintptr_t scalars_length;
} ScalarBatchElement;

typedef struct ScalarResults {
  struct ScalarBatchElement *scalar_batch_elements;
  uintptr_t scalar_batch_elements_length;
} ScalarResults;

#define KeyValue_VT_KEY 4

#define KeyValue_VT_VALUE 6

#define TrtTable_VT_DICT 4

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
__declspec(dllimport) extern const uint32_t ERROR_BUFFER_SIZE;
#else
extern const uint32_t ERROR_BUFFER_SIZE;
#endif

#ifdef _WIN32
__declspec(dllimport) extern const char NULL_BYTE[1];
#else
extern const char NULL_BYTE[1];
#endif

/**
 * Sets the log level. Allowed values (sorted ascending by verbosity): ERROR, WARN, INFO, DEBUG, TRACE
 */
enum DenkflowResult set_log_level(const char *log_level);

/**
 * Print the current DENKflow API version to stdout
 */
enum DenkflowResult print_version(void);

/**
 * Write the current DENKflow API version into variables. "pre_release" must reserve 32 bytes.
 */
enum DenkflowResult get_version(uint32_t *major,
                                uint32_t *minor,
                                uint32_t *patch,
                                char *pre_release);

/**
 * This will write the last error message into a buffer as a NULL-terminated C-string. The buffer must be exactly ERROR_BUFFER_SIZE bytes long.
 */
void get_last_error(char *buffer);

enum DenkflowResult hub_license_source_from_pat(struct HubLicenseSource **hub_license_source,
                                                const char *pat,
                                                const char *license_id,
                                                const char *endpoint);

enum DenkflowResult hub_license_source_free(struct HubLicenseSource **hub_license_source);

enum DenkflowResult hub_license_source_to_one_time_license_source(struct OneTimeLicenseSource **one_time_license_source,
                                                                  struct HubLicenseSource **hub_license_source);

enum DenkflowResult one_time_license_source_free(struct OneTimeLicenseSource **one_time_license_source);

enum DenkflowResult one_time_license_refresh(struct OneTimeLicenseSource *one_time_license_source);

enum DenkflowResult list_all_objects(void);

enum DenkflowResult free_object(void **double_pointer);

enum DenkflowResult free_all_objects(void);

enum DenkflowResult initialize_pipeline(struct InitializedPipeline **initialized_pipeline,
                                        struct Pipeline **pipeline);

enum DenkflowResult initialized_pipeline_free(struct InitializedPipeline **initialized_pipeline);

/**
 * Returns a list of topics.
 */
enum DenkflowResult initialized_pipeline_get_topics(struct TopicInformationArray **topic_information_array,
                                                    struct InitializedPipeline *initialized_pipeline);

enum DenkflowResult initialized_pipeline_publish_image_tensor(struct InitializedPipeline *initialized_pipeline,
                                                              const char *topic,
                                                              struct ImageTensor **message);

enum DenkflowResult initialized_pipeline_run(struct InitializedPipeline *initialized_pipeline,
                                             int64_t timeout_ms);

enum DenkflowResult initialized_pipeline_start(struct InitializedPipeline *initialized_pipeline,
                                               int64_t single_run_timeout_ms);

enum DenkflowResult initialized_pipeline_cancel(struct InitializedPipeline *initialized_pipeline);

enum DenkflowResult pipeline_change_const_tensor_node(struct Pipeline *pipeline,
                                                      const char *node_name,
                                                      struct ConstTensorNodeParams parameters);

enum DenkflowResult initialized_pipeline_change_initialized_const_tensor_node(struct InitializedPipeline *initialized_pipeline,
                                                                              const char *node_name,
                                                                              struct InitializedConstTensorNodeParams parameters);

enum DenkflowResult pipeline_change_virtual_camera_node(struct Pipeline *pipeline,
                                                        const char *node_name,
                                                        struct VirtualCameraNodeParams parameters);

enum DenkflowResult initialized_pipeline_change_initialized_virtual_camera_node(struct InitializedPipeline *initialized_pipeline,
                                                                                const char *node_name,
                                                                                struct InitializedVirtualCameraNodeParams parameters);

enum DenkflowResult pipeline_new(struct Pipeline **pipeline);

enum DenkflowResult pipeline_from_denkflow(struct Pipeline **pipeline,
                                           const char *filename,
                                           void *license_source);

enum DenkflowResult pipeline_free(struct Pipeline **pipeline);

/**
 * Set the number of intra-op threads for ONNX session execution
 *
 * Controls the number of threads used to parallelize execution within nodes/operators.
 * Default is 4.
 *
 * # Arguments
 * * `pipeline` - Pointer to the pipeline
 * * `intra_threads` - Number of intra-op threads (must be > 0)
 */
enum DenkflowResult pipeline_with_intra_threads(struct Pipeline *pipeline, uintptr_t intra_threads);

/**
 * Set the number of inter-op threads for ONNX session execution
 *
 * Controls the number of threads used to parallelize execution between independent nodes/operators.
 * Default is 4.
 *
 * # Arguments
 * * `pipeline` - Pointer to the pipeline
 * * `inter_threads` - Number of inter-op threads (must be > 0)
 */
enum DenkflowResult pipeline_with_inter_threads(struct Pipeline *pipeline, uintptr_t inter_threads);

/**
 * Get the names of all nodes in the pipeline
 */
enum DenkflowResult pipeline_get_node_names(struct StringArray **node_names,
                                            struct Pipeline *pipeline);

/**
 * Sets the channel size for the topics of this pipeline (default: 1, minimum: 1)
 */
enum DenkflowResult pipeline_set_channel_size(struct Pipeline *pipeline, uintptr_t channel_size);

/**
 * Sets the session info for a specific node by name, or for all nodes (if node_name is a single null-byte)
 * Valid execution_provider values are (not case-sensitive): CPU, CUDA, DirectML, TensorRT
 */
enum DenkflowResult pipeline_set_session_info(struct Pipeline *pipeline,
                                              const char *execution_provider,
                                              int32_t device_id,
                                              const char *node_name);

enum DenkflowResult pipeline_add_bounding_box_filter_node(struct BoundingBoxFilterNodeReference **bounding_box_filter_node_reference,
                                                          struct Pipeline *pipeline,
                                                          const char *node_name,
                                                          const char *bounding_boxes_topic,
                                                          const char *iou_threshold_topic,
                                                          const char *score_threshold_topic,
                                                          const char *execution_provider,
                                                          int32_t device_id);

enum DenkflowResult pipeline_add_const_tensor_node_int(struct ConstTensorNodeReference **const_tensor_node_reference,
                                                       struct Pipeline *pipeline,
                                                       const char *node_name,
                                                       const int64_t *array,
                                                       uintptr_t array_length);

enum DenkflowResult pipeline_add_image_anomaly_detection_node(struct ImageAnomalyDetectionNodeReference **image_anomaly_detection_node_reference,
                                                              struct Pipeline *pipeline,
                                                              const char *node_name,
                                                              const char *image_topic,
                                                              const char *model_path,
                                                              void *license_source,
                                                              const char *execution_provider,
                                                              int32_t device_id);

enum DenkflowResult pipeline_add_image_classification_node(struct ImageClassificationNodeReference **image_classification_node_reference,
                                                           struct Pipeline *pipeline,
                                                           const char *node_name,
                                                           const char *image_topic,
                                                           const char *model_path,
                                                           void *license_source,
                                                           const char *execution_provider,
                                                           int32_t device_id);

enum DenkflowResult pipeline_add_image_instance_segmentation_node(struct ImageInstanceSegmentationNodeReference **image_instance_segmentation_node_reference,
                                                                  struct Pipeline *pipeline,
                                                                  const char *node_name,
                                                                  const char *image_topic,
                                                                  const char *model_path,
                                                                  void *license_source,
                                                                  const char *execution_provider,
                                                                  int32_t device_id);

enum DenkflowResult pipeline_add_image_object_detection_node(struct ImageObjectDetectionNodeReference **image_object_detection_node_reference,
                                                             struct Pipeline *pipeline,
                                                             const char *node_name,
                                                             const char *image_topic,
                                                             const char *model_path,
                                                             void *license_source,
                                                             const char *execution_provider,
                                                             int32_t device_id);

enum DenkflowResult pipeline_add_image_patches_node(struct ImagePatchesNodeReference **image_patches_node_reference,
                                                    struct Pipeline *pipeline,
                                                    const char *node_name,
                                                    const char *image_topic,
                                                    const char *bounding_boxes_topic,
                                                    const char *target_size_topic,
                                                    enum EnumImagePatchesNodeResizeMode resize_mode,
                                                    const char *execution_provider,
                                                    int32_t device_id);

enum DenkflowResult pipeline_add_image_resize_node(struct ImageResizeNodeReference **image_resize_node_reference,
                                                   struct Pipeline *pipeline,
                                                   const char *node_name,
                                                   const char *image_topic,
                                                   const char *target_size_topic,
                                                   const char *execution_provider,
                                                   int32_t device_id);

enum DenkflowResult pipeline_add_image_segmentation_node(struct ImageSegmentationNodeReference **image_segmentation_node_reference,
                                                         struct Pipeline *pipeline,
                                                         const char *node_name,
                                                         const char *image_topic,
                                                         const char *model_path,
                                                         void *license_source,
                                                         const char *execution_provider,
                                                         int32_t device_id);

enum DenkflowResult pipeline_add_ocr_node(struct OCRNodeReference **ocr_node_reference,
                                          struct Pipeline *pipeline,
                                          const char *node_name,
                                          const char *image_topic,
                                          const char *model_path,
                                          void *license_source,
                                          const char *execution_provider,
                                          int32_t device_id);

enum DenkflowResult pipeline_add_virtual_camera_node(struct VirtualCameraNodeReference **virtual_camera_node_reference,
                                                     struct Pipeline *pipeline,
                                                     const char *node_name,
                                                     const char *folder_path);

enum DenkflowResult receiver_receive_bounding_box_tensor(struct BoundingBoxTensor **bounding_box_tensor,
                                                         Receiver_Tensor *receiver);

enum DenkflowResult bounding_box_tensor_free(struct BoundingBoxTensor **bounding_box_tensor);

enum DenkflowResult bounding_box_tensor_to_objects(struct BoundingBoxResults **bounding_box_results,
                                                   struct BoundingBoxTensor *bounding_box_tensor,
                                                   float confidence_threshold);

enum DenkflowResult bounding_box_results_free(struct BoundingBoxResults **bounding_box_results);

enum DenkflowResult image_tensor_from_file(struct ImageTensor **image_tensor, const char *filename);

enum DenkflowResult image_tensor_from_files(struct ImageTensor **image_tensor,
                                            const char *const *filenames,
                                            uintptr_t filenames_length);

enum DenkflowResult image_tensor_from_image_data(struct ImageTensor **image_tensor,
                                                 uintptr_t image_width,
                                                 uintptr_t image_height,
                                                 uintptr_t image_channels,
                                                 const char *data,
                                                 uintptr_t data_length);

enum DenkflowResult receiver_receive_image_tensor(struct ImageTensor **image_tensor,
                                                  Receiver_Tensor *receiver);

enum DenkflowResult image_tensor_free(struct ImageTensor **image_tensor);

enum DenkflowResult receiver_receive_instance_segmentation_mask_tensor(struct InstanceSegmentationMaskTensor **instance_segmentation_mask_tensor,
                                                                       Receiver_Tensor *receiver);

enum DenkflowResult instance_segmentation_mask_tensor_free(struct InstanceSegmentationMaskTensor **instance_segmentation_mask_tensor);

enum DenkflowResult instance_segmentation_mask_tensor_to_objects(struct SegmentationResults **segmentation_results,
                                                                 struct InstanceSegmentationMaskTensor *instance_segmentation_mask_tensor,
                                                                 float segmentation_threshold,
                                                                 struct BoundingBoxTensor *bounding_box_tensor,
                                                                 float confidence_threshold);

enum DenkflowResult receiver_receive_ocr_tensor(struct OcrTensor **ocr_tensor,
                                                Receiver_Tensor *receiver);

enum DenkflowResult ocr_tensor_free(struct OcrTensor **ocr_tensor);

enum DenkflowResult ocr_tensor_to_objects(struct OcrResults **ocr_results,
                                          struct OcrTensor *ocr_tensor);

enum DenkflowResult ocr_results_free(struct OcrResults **ocr_results);

enum DenkflowResult initialized_pipeline_subscribe(Receiver_Tensor **receiver,
                                                   struct InitializedPipeline *initialized_pipeline,
                                                   const char *topic);

enum DenkflowResult receiver_receive_scalar_tensor(struct ScalarTensor **scalar_tensor,
                                                   Receiver_Tensor *receiver);

enum DenkflowResult scalar_tensor_free(struct ScalarTensor **scalar_tensor);

enum DenkflowResult scalar_tensor_to_objects(struct ScalarResults **scalar_results,
                                             struct ScalarTensor *scalar_tensor);

enum DenkflowResult scalar_results_free(struct ScalarTensor **scalar_results);

enum DenkflowResult receiver_receive_segmentation_mask_tensor(struct SegmentationMaskTensor **segmentation_mask_tensor,
                                                              Receiver_Tensor *receiver);

enum DenkflowResult segmentation_mask_tensor_free(struct SegmentationMaskTensor **segmentation_mask_tensor);

enum DenkflowResult segmentation_mask_tensor_to_objects(struct SegmentationResults **segmentation_results,
                                                        struct SegmentationMaskTensor *segmentation_mask_tensor,
                                                        float segmentation_threshold);

enum DenkflowResult segmentation_results_free(struct SegmentationResults **segmentation_results);

enum DenkflowResult receiver_receive_tensor(struct BaseTensor **tensor, Receiver_Tensor *receiver);

enum DenkflowResult tensor_free(struct BaseTensor **tensor);

enum DenkflowResult tensor_from_int64_array(struct BaseTensor **tensor,
                                            const int64_t *data,
                                            const uintptr_t *shape,
                                            uintptr_t shape_length);

enum DenkflowResult tensor_from_uint64_array(struct BaseTensor **tensor,
                                             const uint64_t *data,
                                             const uintptr_t *shape,
                                             uintptr_t shape_length);

enum DenkflowResult tensor_from_double_array(struct BaseTensor **tensor,
                                             const double *data,
                                             const uintptr_t *shape,
                                             uintptr_t shape_length);

#ifdef __cplusplus
}
#endif
