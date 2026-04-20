#include <cstdint>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <oryx/crt/stopwatch.hpp>
#include <oryx/crt/argparse.hpp>

#include <denkflow.h>

using namespace oryx;

constexpr std::string_view kInputTopic = "camera/image";
constexpr std::string_view kOutputTopic = "bounding_box_filter_node/filtered_bounding_boxes";
constexpr std::string_view kOutFile = "out.png";
constexpr auto kConfidenceTreshold = 0.5F;

constexpr int kFontFace{cv::FONT_HERSHEY_SIMPLEX};
constexpr int kLineType{cv::LINE_8};
constexpr double kFontScale{0.5};
const cv::Scalar kColor{0, 0, 255};

void PrintDnkError(enum DenkflowResult error_code, const char* function_name) {
    std::cout << function_name << ": " << error_code << "\n";
    char error_buffer[ERROR_BUFFER_SIZE];
    memset(error_buffer, 0, ERROR_BUFFER_SIZE);
    get_last_error(error_buffer);
    std::cout << " (" << static_cast<char*>(error_buffer) << ")" << "\n";
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
    cv::putText(img, text, point, kFontFace, kFontScale, kColor, kLineType);
}

auto main(int argc, char* argv[]) -> int {
    Pipeline* pipeline{};
    InitializedPipeline* initialized_pipeline{};
    ImageTensor* image_tensor{};
    Receiver_Tensor* receiver{};
    BoundingBoxTensor* tensor{};
    BoundingBoxResults* results{};
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

    DenkflowResult r;
    r = hub_license_source_from_pat(&hub_license_source, pat.value().c_str(), NULL_BYTE, NULL_BYTE);
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

    r = initialized_pipeline_subscribe(&receiver, initialized_pipeline, kOutputTopic.data());
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
    std::cout << "\tWidth: " << chw.cols << "\n";
    std::cout << "\tHeight: " << chw.rows << "\n";
    std::cout << "\tColor Channels: " << chw.channels() << "\n";
    std::cout << "\tWidth: " << chw.cols << "\n";
    std::cout << "\tBytes: " << size_bytes << "\n";

    // FIX_ME: Using image tensor from file yields detection results switchting to image_tensor_from_data does not yield
    // any detections.
    //  r = image_tensor_from_file(&image_tensor, input.value().c_str());
    r = image_tensor_from_image_data(&image_tensor, chw.cols, chw.rows, chw.channels(),
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

    r = receiver_receive_bounding_box_tensor(&tensor, receiver);
    if (r != DenkflowResult_Ok) {
        PrintDnkError(r, "initialized_pipeline_run");
        return 1;
    }

    r = bounding_box_tensor_to_objects(&results, tensor, kConfidenceTreshold);
    if (r != DenkflowResult_Ok) {
        PrintDnkError(r, "bounding_box_tensor_to_objects");
        return 1;
    }

    auto elapsed = sw.ElapsedMs();
    std::cout << "Pipeline took: " << elapsed.count() << " ms\n";

    std::cout << "Detection Results: \n";
    if (results) {
        for (uintptr_t b = 0; b < results->bounding_boxes_length; b++) {
            auto& bbox = results->bounding_boxes[b];

            std::cout << bbox.class_label.name << " :" << bbox.confidence << "\n";
            DrawBox(image, ToCvRect(bbox, image.cols, image.rows));
        }
    }

    cv::imwrite(std::string(kOutFile), image);
    std::cout << "Annotation saved to: " << kOutFile << "\n";

    free_object((void**)&tensor);
    free_object((void**)&results);
    free_object((void**)&hub_license_source);
    free_object((void**)&initialized_pipeline);
    free_object((void**)&receiver);
    return 0;
}