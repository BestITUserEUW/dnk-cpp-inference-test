#include <cstdint>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <oryx/crt/stopwatch.hpp>
#include <oryx/crt/argparse.hpp>
#include <oryx/crt/scope_exit.hpp>
#include <oryx/crt/enchantum.hpp>

#include <denkflow.h>

using namespace oryx;

enum class InferenceType : std::uint8_t { detect, classify };

constexpr std::string_view kInputTopic = "camera/image";
constexpr std::string_view kBboxTopic = "bounding_box_filter_node/filtered_bounding_boxes";
constexpr std::string_view kClassifyTopic = "classification_node/output";
constexpr std::string_view kOutFile = "out.png";
constexpr auto kConfidenceTreshold = 0.2F;

constexpr int kFontFace{cv::FONT_HERSHEY_SIMPLEX};
constexpr int kLineType{cv::LINE_8};
constexpr double kFontScale{0.5};
const cv::Scalar kColor{0, 0, 255};

void PrintDnkError(enum DenkflowResult error_code, const char* function_name) {
    std::cout << function_name << ": " << error_code << "\n";
    auto buffer = static_cast<char*>(calloc(ERROR_BUFFER_SIZE, sizeof(char)));
    get_last_error(buffer);
    std::cout << " (" << buffer << ")" << "\n";
    free(buffer);
}

auto ToCvRect(const BoundingBox& box, int img_width, int img_height) -> cv::Rect {
    float x1 = box.x1 * img_width;
    float y1 = box.y1 * img_height;
    float x2 = box.x2 * img_width;
    float y2 = box.y2 * img_height;

    int x = static_cast<int>(std::round(std::min(x1, x2)));
    int y = static_cast<int>(std::round(std::min(y1, y2)));
    int width = static_cast<int>(std::round(std::abs(x2 - x1)));
    int height = static_cast<int>(std::round(std::abs(y2 - y1)));

    return {x, y, width, height};
}

void DrawBox(const cv::Mat& img, const cv::Rect& box) { cv::rectangle(img, box, kColor, 1, kLineType); }

void DrawText(const cv::Mat& img, const std::string& text, const cv::Point& point) {
    cv::putText(img, text, point, kFontFace, kFontScale, kColor, 1, kLineType);
}

auto PostProcessObjectDetection(Receiver_Tensor* receiver, const cv::Mat& image) -> bool {
    BoundingBoxTensor* tensor{};
    BoundingBoxResults* results{};

    crt::ScopeExit se{[&tensor, &results] {
        if (tensor) free_object((void**)&tensor);
        if (results) free_object((void**)&results);
    }};

    DenkflowResult r = receiver_receive_bounding_box_tensor(&tensor, receiver);
    if (r != DenkflowResult_Ok) {
        PrintDnkError(r, "initialized_pipeline_run");
        return false;
    }

    r = bounding_box_tensor_to_objects(&results, tensor, kConfidenceTreshold);
    if (r != DenkflowResult_Ok) {
        PrintDnkError(r, "bounding_box_tensor_to_objects");
        return false;
    }

    std::cout << "ObjectDetection results: \n";
    if (results) {
        for (uintptr_t b = 0; b < results->bounding_boxes_length; b++) {
            auto& bbox = results->bounding_boxes[b];

            std::cout << "\t" << bbox.class_label.name << ": " << bbox.confidence << "\n";
            DrawBox(image, ToCvRect(bbox, image.cols, image.rows));
        }
    }
    return true;
}

auto PostProcessClassification(Receiver_Tensor* receiver, const cv::Mat& image) {
    ScalarTensor* tensor{};
    ScalarResults* results{};

    crt::ScopeExit se{[&tensor, &results] {
        if (tensor) free_object((void**)&tensor);
        if (results) free_object((void**)&results);
    }};

    DenkflowResult r = receiver_receive_scalar_tensor(&tensor, receiver);
    if (r != DenkflowResult_Ok) {
        PrintDnkError(r, "receiver_receive_scalar_tensor");
        return false;
    }

    r = scalar_tensor_to_objects(&results, tensor);
    if (r != DenkflowResult_Ok) {
        PrintDnkError(r, "scalar_tensor_to_objects");
        return false;
    }

    std::cout << "Classification results: \n";
    int y_off = 20;
    if (results) {
        for (uintptr_t b = 0; b < results->scalar_batch_elements_length; b++) {
            const auto& element = results->scalar_batch_elements[b];

            for (uintptr_t c = 0; c < element.scalars_length; c++) {
                const auto& scalar = element.scalars[c];

                std::cout << "\t" << scalar.class_label.name << ": " << scalar.value << "\n";
                DrawText(image, std::format("{}: {}", scalar.class_label.name, scalar.value), cv::Point(0, y_off));
                y_off += 20;
            }
        }
    }
    return true;
}

auto main(int argc, char* argv[]) -> int {
    Pipeline* pipeline{};
    InitializedPipeline* initialized_pipeline{};
    ImageTensor* image_tensor{};
    Receiver_Tensor* receiver{};
    HubLicenseSource* hub_license_source{};

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    set_log_level("INFO");

    auto parser = crt::ArgumentParser(argc, argv);
    auto pat = parser.GetValue<std::string>("--pat");
    if (!pat) {
        std::cout << "--pat is required!" << "\n";
        return 1;
    }

    auto model = parser.GetValue<std::string>("--model");
    if (!model) {
        std::cout << "--model is required!" << "\n";
        return 1;
    }

    auto input = parser.GetValue<std::string>("--input");
    if (!input) {
        std::cout << "--input is required!" << "\n";
        return 1;
    }

    auto type_str = parser.GetValue<std::string>("--type");
    if (!type_str) {
        std::cout << "--type is required (detect, classify)" << "\n";
        return 1;
    }

    auto type = enchantum::cast<InferenceType>(type_str.value());
    if (!type) {
        std::cout << "Invalid inference type passed!" << "\n";
        return 1;
    }

    DenkflowResult r = hub_license_source_from_pat(&hub_license_source, pat.value().c_str(), NULL_BYTE, NULL_BYTE);
    if (r != DenkflowResult_Ok) {
        PrintDnkError(r, "hub_license_source_from_pat");
        return 1;
    }

    r = pipeline_from_denkflow(&pipeline, model.value().c_str(), hub_license_source);
    if (r != DenkflowResult_Ok) {
        PrintDnkError(r, "pipeline_from_denkflow");
        return 1;
    }

    pipeline_with_intra_threads(pipeline, 16);  // Threads for parallelism within operators
    pipeline_with_inter_threads(pipeline, 4);   // Threads for parallelism between operators

    r = initialize_pipeline(&initialized_pipeline, &pipeline);
    if (r != DenkflowResult_Ok) {
        PrintDnkError(r, "initialize_pipeline");
        return 1;
    }

    r = initialized_pipeline_subscribe(
        &receiver, initialized_pipeline,
        type.value() == InferenceType::detect ? kBboxTopic.data() : kClassifyTopic.data());
    if (r != DenkflowResult_Ok) {
        PrintDnkError(r, "initialized_pipeline_subscribe");
        return 1;
    }

    auto image = cv::imread(input.value());

    std::vector<cv::Mat> channels(image.channels());
    cv::split(image, channels);
    cv::Mat chw;
    cv::vconcat(channels, chw);

    auto size_bytes = chw.step[0] * chw.rows;
    std::cout << "Image Meta: \n";
    std::cout << "\tWidth: " << image.cols << "\n";
    std::cout << "\tHeight: " << image.rows << "\n";
    std::cout << "\tColor Channels: " << image.channels() << "\n";
    std::cout << "\tWidth: " << image.cols << "\n";
    std::cout << "\tBytes: " << size_bytes << "\n";

    r = image_tensor_from_image_data(&image_tensor, image.cols, image.rows, image.channels(),
                                     reinterpret_cast<char*>(chw.data), size_bytes);
    if (r != DenkflowResult_Ok) {
        PrintDnkError(r, "image_tensor_from_file");
        return 1;
    }

    crt::Stopwatch sw{};
    r = initialized_pipeline_publish_image_tensor(initialized_pipeline, kInputTopic.data(), &image_tensor);
    if (r != DenkflowResult_Ok) {
        PrintDnkError(r, "initialized_pipeline_publish_image_tensor");
        return 1;
    }

    r = initialized_pipeline_run(initialized_pipeline, 8000);
    if (r != DenkflowResult_Ok) {
        PrintDnkError(r, "initialized_pipeline_run");
        return 1;
    }

    if (type.value() == InferenceType::detect) {
        PostProcessObjectDetection(receiver, image);
    } else {
        PostProcessClassification(receiver, image);
    }

    auto elapsed = sw.ElapsedMs();
    std::cout << "Pipeline took: " << elapsed.count() << " ms\n";

    cv::imwrite(std::string(kOutFile), image);
    std::cout << "Annotation saved to: " << kOutFile << "\n";

    free_object((void**)&hub_license_source);
    free_object((void**)&initialized_pipeline);
    free_object((void**)&receiver);
    return 0;
}