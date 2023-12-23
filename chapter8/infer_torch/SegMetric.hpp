#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

class SegMetric
{
private:
    std::vector<size_t> intersection_counts;
    std::vector<size_t> union_counts;

public:
    std::vector<float> compute_IoUs()
    {
        std::vector<float> IoUs;
        for (size_t i = 0; i < intersection_counts.size(); i++)
        {
            if (union_counts[i] > 0)
            {
                float iou = (float)intersection_counts[i] / (float)union_counts[i];
                IoUs.push_back(iou);
            }
            else
            {
                IoUs.push_back(-1);
            }
        }

        return IoUs;
    }

    std::pair<int, int> count_IU(const cv::Mat &pred, const cv::Mat &target, uint8_t cat)
    {
        auto pred_cat = (pred == cat);
        auto target_cat = (target == cat);
        cv::Mat intersection_mask;
        cv::bitwise_and(pred_cat, target_cat, intersection_mask);
        cv::Mat union_mask = (pred_cat + target_cat);
        int intersection = cv::countNonZero(intersection_mask);
        int uni = cv::countNonZero(union_mask);
        return {intersection, uni};
    }

    float compute_mIoU(std::vector<int> cates)
    {
        float iou = 0;
        float valid_class = 0;
        auto IoUs = compute_IoUs();
        if (cates.size() == 0)
        {
            for (int i = 0; i < IoUs.size(); i++)
                cates.push_back(i);
        }

        for (const int &i : cates)
        {
            if (IoUs[i] != -1)
            {
                iou += IoUs[i];
                valid_class++;
            }
        }
        return iou / valid_class;
    }

    void append(const cv::Mat &pred, const cv::Mat &target, int num_classes)
    {
        if (intersection_counts.size() == 0)
        {
            intersection_counts.resize(num_classes, 0);
            union_counts.resize(num_classes, 0);
        }
        for (int i = 0; i < num_classes; i++)
        {
            auto iu = count_IU(pred, target, i);
            intersection_counts[i] += iu.first;
            union_counts[i] += iu.second;
        }
    }

    void dump_counts(std::string file)
    {
        auto ious = compute_IoUs();
        std::ofstream ofs(file);
        for (int i = 0; i < ious.size(); i++)
        {
            ofs << ious[i] << "\n";
        }
        ofs.close();
    }

    float compute_IoU(const cv::Mat &pred, const cv::Mat &target, uint8_t cat)
    {
        auto iu = count_IU(pred, target, cat);
        int intersection = iu.first;
        int uni = iu.second;
        if ((uni == 0) || (intersection == 0))
        {
            return -1;
        }
        else
        {
            return (float)intersection / (float)uni;
        }
    }
};