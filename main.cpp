#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "stdc++.h"
#include "utility.hpp"

using namespace std;

#define WIDTH (testFaxImg.cols)
#define HEIGHT (testFaxImg.rows)
#define ARRLEN 257

cv::Mat testFaxImg = cv::imread("../img/pic1.jpg", 0);

// test matrix
cv::Mat testMat = (cv::Mat_<uchar>(8, 8) << 255, 255, 255, 255, 255, 255, 255, 0,
                   255, 255, 255, 255, 255, 255, 255, 255,
                   0, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 0,
                   0, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 0, 255, 0);

//// generate code table

int freq[257] = {0},                         // 信源符号V的出现频次
    codeSize[257] = {0},                     // 信源符号V的码长
    others[257] = {[0 ... ARRLEN - 1] = -1}, // 编码树中当前分支的下一个信源符号的索引号
    bits[32],                                // 霍夫曼码字长度为 n+1 的码字数量
    huffVal[183],                            // 按简表 bits 给出的码字长度的数量，按码字长度从小到大的顺序，
    // 码字长度短的同长度码字优先，依据由 freq 算出的 codeSize[V]值，按 V 从小到大的顺序排列。
    huffSize[183], // 保存将生成的霍夫曼码字长度
    huffCode[183],
    eHuffCode[257],
    eHuffSize[257];

int lastK;

string encodeIdx = "", decodeIdx = "";

void writeFaxImg(const cv::Mat &faxImg, string path = "../faxImg.txt")
{
    cv::threshold(faxImg, faxImg, 200, 255, cv::THRESH_BINARY);
    ofstream fout(path);
    // #pragma omp parallel for
    for (int i = 0; i < faxImg.rows; i++)
    {
        for (int j = 0; j < faxImg.cols; j++)
        {
            if ((int)faxImg.at<uchar>(i, j) == 255)
                fout << 1 << " ";
            else
                fout << 0 << " ";
        }
        fout << endl;
    }
    fout.close();
}

vector<bool> mat2Bits(const cv::Mat &Img)
{
    vector<bool> vec;
    // #pragma omp parallel for
    for (int i = 0; i < Img.rows; i++)
    {
        for (int j = 0; j < Img.cols; j++)
        {
            if ((int)Img.at<uchar>(i, j) == 255)
                vec.push_back(1);
            else
                vec.push_back(0);
        }
    }
    return vec;
}

cv::Mat bits2Mat(const vector<bool> &input, int row, int col)
{
    cv::Mat output(row, col, CV_8UC1);
    // #pragma omp parallel for
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; ++j)
        {
            output.at<uchar>(i, j) = input[i * col + j] ? 255 : 0;
        }
    }
    return output;
}

vector<bool> int2Bits(unsigned int num, unsigned int len)
{
    vector<bool> bits;
    for (int i = 0; i < len; i++)
    {
        bits.push_back(num % 2);
        num /= 2;
    }
    reverse(bits.begin(), bits.end());
    return bits;
}

unsigned int RL2Idx(unsigned int i, bool isWhite)
{
    if (0 <= i && i <= 63 && isWhite)
        return i;
    else if (0 <= i && i <= 63 && !isWhite)
        return i + 64;
    else if (64 <= i && i <= 1728 && i % 64 == 0)
    {
        if (isWhite)
            return 127 + i / 64;
        else
            return 154 + i / 64;
    }
    else
        return 255;
}

vector<bool> idx2Bits(unsigned int idx)
{
    vector<bool> bits;
    if (0 <= idx && idx <= 63)
    {
        bits = vector<bool>(idx, 1);
    }
    else if (64 <= idx && idx <= 127)
    {
        bits = vector<bool>(idx - 64, 0);
    }
    else if (128 <= idx && idx <= 154)
    {
        bits = vector<bool>((idx - 127) * 64, 1);
    }
    else if (155 <= idx && idx <= 181)
    {
        bits = vector<bool>((idx - 154) * 64, 0);
    }
    else if (idx == 182)
    {
    }
    return bits;
}

void calcRLFreq(cv::Mat &m)
{
    bool isWhite, isWhitePre, compRes;
    unsigned int cnt;
    freq[256] = 1;
    for (int i = 0; i < m.rows; i++)
    {
        isWhite = false, isWhitePre = true, compRes = false;
        cnt = 0;
        freq[0]++;
        freq[182]++;
        for (int j = 0; j < m.cols; j++)
        {
            isWhite = (m.at<uchar>(i, j) == 255);
            compRes = (isWhite ^ isWhitePre) ? false : true;
            if (compRes) // calc white RL
            {
                cnt++;
            }
            else
            {
                if (cnt > 64)
                {
                    int remain = cnt % 64;
                    freq[RL2Idx(cnt - remain, isWhitePre)]++;
                    freq[RL2Idx(remain, isWhitePre)]++;
                }
                else
                {
                    freq[RL2Idx(cnt, isWhitePre)]++;
                }
                cnt = 1;
            }
            isWhitePre = isWhite;
        }
        if (cnt > 64)
        {
            int remain = cnt % 64;
            freq[RL2Idx(cnt - remain, isWhitePre)]++;
            freq[RL2Idx(remain, isWhitePre)]++;
        }
        else
        {
            freq[RL2Idx(cnt, isWhitePre)]++;
        }
    }
}

void calcHuffmanLen()
{
    auto stack = desTupVec(freq, ARRLEN);
    tuple<size_t, int> idxV1, idxV2;
    while (!stack.empty())
    {
        sort(stack.begin(), stack.end(),
             [](const tuple<size_t, int> &a,
                const tuple<size_t, int> &b)
             { return (get<1>(a) > get<1>(b)); });
        idxV1 = stack.back();
        if (get<1>(idxV1) == 0)
        {
            stack.pop_back();
            continue;
        }
        idxV2 = stack.end()[-2];
        get<1>(stack.end()[-2]) = get<1>(idxV1) + get<1>(idxV2);
        // cout << "(" << get<0>(idxV1) << ", " << get<0>(idxV2) << ")" << endl;
        stack.pop_back();
        if (stack.empty())
            break;

        codeSize[get<0>(idxV2)]++;
        while (others[get<0>(idxV2)] != -1)
        {
            get<0>(idxV2) = others[get<0>(idxV2)];
            codeSize[get<0>(idxV2)]++;
        }
        others[get<0>(idxV2)] = get<0>(idxV1);
        // cout << "(" << get<0>(idxV1) << ", " << get<0>(idxV2) << ")" << endl;
        codeSize[get<0>(idxV1)]++;
        while (others[get<0>(idxV1)] != -1)
        {
            get<0>(idxV1) = others[get<0>(idxV1)];
            codeSize[get<0>(idxV1)]++;
        }
    }
}

void cntBits()
{
    int i = 0;
    while (i != 257)
    {
        if (codeSize[i] != 0)
            bits[codeSize[i]]++;
        i++;
    }
}

void adjustBits()
{
    int i = 31, j;
    while (1)
    {
        if (bits[i] > 0)
        {
            j = i - 1;
            j--;
            while (bits[j] <= 0)
                j--;
            bits[i] = bits[i] - 2;
            bits[i - 1] = bits[i - 1] + 1;
            bits[j + 1] = bits[j + 1] + 2;
            bits[j] = bits[j] - 1;
        }
        else
        {
            i--;
            if (i != 14)
                continue;
            while (bits[i] == 0)
                i--;
            bits[i] = bits[i] - 1;
            break;
        }
    }
}

void sortInput()
{
    int i = 1, k = 0, j;
    do
    {
        j = 0;
        do
        {
            if (codeSize[j] == i)
            {
                huffVal[k] = j;
                k++;
            }
            j++;
        } while (j <= 255);
        i++;
    } while (i <= 32);
}

void genSizeTable()
{
    int k = 0, i = 1, j = 1;
    do
    {
        while (j <= bits[i])
        {
            huffSize[k] = i;
            k++;
            j++;
        }
        i++;
        j = 1;
    } while (i <= 14);
    huffSize[k] = 0;
    lastK = k;
}

void genCodeTable()
{
    int k = 0, code = 0, si = huffSize[0];
    do
    {
        do
        {
            huffCode[k] = code;
            code++;
            k++;
        } while (huffSize[k] == si);
        if (huffSize[k] == 0)
        {
            return;
        }
        else
        {
            do
            {
                code = code << 1;
                si++;
            } while (huffSize[k] != si);
        }
    } while (huffSize[k] == si);
}

void orderCodes()
{
    int k = 0, i;
    do
    {
        i = huffVal[k];
        eHuffCode[i] = huffCode[k];
        eHuffSize[i] = huffSize[k];
        k++;
    } while (k < lastK);
}

//// Decode
int valPtr[17], minCode[17], maxCode[17];
void genDecodeTable()
{
    int i = 0, j = 0;
    while (1)
    {
        while (1)
        {
            i++;
            if (i > 14)
            {
                return;
            }
            else
            {
                if (bits[i] == 0)
                {
                    maxCode[i] = -1;
                    continue;
                }
                else
                {
                    valPtr[i] = j;
                    minCode[i] = huffCode[j];
                    j = j + bits[i] - 1;
                    maxCode[i] = huffCode[j];
                    j++;
                    break;
                }
            }
        }
    }
}

//// Process
void preprocess()
{
    cv::Mat img;
    // #pragma omp parallel for
    for (int i = 1; i <= 3; i++)
    {
        img = cv::imread("../img/pic" + to_string(i) + ".jpg", 0);
        cv::threshold(img, img, 200, 255, cv::THRESH_BINARY);
        calcRLFreq(img);
    }
    calcHuffmanLen();
    cntBits();
    adjustBits();
    sortInput();
    genSizeTable();
    genCodeTable();
    orderCodes();
    // gen decode table
    genDecodeTable();
}

vector<bool> encode(const cv::Mat &m)
{
    bool isWhite, isWhitePre, compRes;
    unsigned int cnt;
    unsigned int idx1, idx2;
    vector<bool> output;
    for (int i = 0; i < m.rows; i++)
    {
        isWhite = false, isWhitePre = true, compRes = false;
        cnt = 0;
        // cout << "i row:" << i <<endl;
        for (int j = 0; j < m.cols; j++)
        {
            isWhite = (m.at<uchar>(i, j) == 255);
            compRes = (isWhite ^ isWhitePre) ? false : true;
            if (compRes) // calc white RL
            {
                cnt++;
            }
            else
            {
                if (cnt > 64)
                {
                    int remain = cnt % 64;
                    idx1 = RL2Idx(cnt - remain, isWhitePre);
                    idx2 = RL2Idx(remain, isWhitePre);
                    encodeIdx += to_string(idx1) + " ";
                    encodeIdx += to_string(idx2) + " ";
                    if (eHuffSize[idx1] == 0 || eHuffSize[idx2] == 0)
                    {
                        cerr << "Encode Error: eHuffSize[idx] == 0" << endl;
                        cerr << "idx is: " << idx1 << " " << idx2 << endl;
                        break;
                    }
                    auto tmp = int2Bits(eHuffCode[idx1], eHuffSize[idx1]);
                    output.insert(output.end(), tmp.begin(), tmp.end());
                    tmp = int2Bits(eHuffCode[idx2], eHuffSize[idx2]);
                    output.insert(output.end(), tmp.begin(), tmp.end());
                }
                else
                {
                    idx1 = RL2Idx(cnt, isWhitePre);
                    encodeIdx += to_string(idx1) + " ";
                    if (eHuffSize[idx1] == 0)
                    {
                        cerr << "Encode Error: eHuffSize[idx] == 0" << endl;
                        cerr << "idx is: " << idx1 << endl;
                        break;
                    }
                    auto tmp = int2Bits(eHuffCode[idx1], eHuffSize[idx1]);
                    output.insert(output.end(), tmp.begin(), tmp.end());
                }
                cnt = 1;
            }
            isWhitePre = isWhite;
        }
        if (cnt > 64)
        {
            int remain = cnt % 64;
            idx1 = RL2Idx(cnt - remain, isWhitePre);
            idx2 = RL2Idx(remain, isWhitePre);
            encodeIdx += to_string(idx1) + " ";
            encodeIdx += to_string(idx2) + " ";
            if (eHuffSize[idx1] == 0 || eHuffSize[idx2] == 0)
            {
                cerr << "Encode Error: eHuffSize[idx] == 0" << endl;
                cerr << "idx is: " << idx1 << " " << idx2 << endl;
                break;
            }
            auto tmp = int2Bits(eHuffCode[idx1], eHuffSize[idx1]);
            output.insert(output.end(), tmp.begin(), tmp.end());
            tmp = int2Bits(eHuffCode[idx2], eHuffSize[idx2]);
            output.insert(output.end(), tmp.begin(), tmp.end());
        }
        else
        {
            idx1 = RL2Idx(cnt, isWhitePre);
            encodeIdx += to_string(idx1) + " ";
            if (eHuffSize[idx1] == 0)
            {
                cerr << "Encode Error: eHuffSize[idx] == 0" << endl;
                cerr << "idx is: " << idx1 << endl;
                break;
            }
            auto tmp = int2Bits(eHuffCode[idx1], eHuffSize[idx1]);
            output.insert(output.end(), tmp.begin(), tmp.end());
        }
    }
    return output;
}

vector<bool> decode(const vector<bool> &input)
{
    int i = 1, k = 0, j;
    int size = input.size();
    vector<bool> output;
    while (k < size)
    {
        long code = input[k];
        i = 1;
        while (code > maxCode[i])
        {
            i++;
            k++;
            code = (code << 1) + input[k];
        }
        j = valPtr[i] + code - minCode[i];
        k++;
        // cout << "end k:" << k << endl;
        auto res = idx2Bits(huffVal[j]);
        decodeIdx += to_string(huffVal[j]) + " ";
        output.insert(output.end(), res.begin(), res.end());
    }
    return output;
}

string filtrateZero(int *arr, int size)
{
    string res = "";
    int i = 0;
    while (i < size)
    {
        if (arr[i] != 0)
        {
            res += "[";
            res += to_string(i);
            res += ":";
            res += to_string(arr[i]);
            res += "]";
        }
        i++;
    }
    res += "\n\n";
    return res;
}

void codeTable2File()
{
    ofstream fout;
    fout.open("../stdout.txt");
    string concat = "";
    concat += "freq:\n";
    concat += filtrateZero(freq, 257);
    concat += "codeSize:\n";
    concat += filtrateZero(codeSize, 257);
    concat += "huffVal:\n";
    concat += filtrateZero(huffVal, 183);
    concat += "huffCode:\n";
    concat += filtrateZero(huffCode, 183);
    concat += "huffSize:\n";
    concat += filtrateZero(huffSize, 183);
    concat += "eHuffCode:\n";
    concat += filtrateZero(eHuffCode, 257);
    concat += "eHuffSize:\n";
    concat += filtrateZero(eHuffSize, 257);
    fout << concat;
    fout << encodeIdx;
    fout << "\n";
    fout << decodeIdx;
    fout.close();
}

int main()
{
    // count time wiht chrono
    auto start = chrono::steady_clock::now();
    preprocess();
    codeTable2File();
    auto end1 = chrono::steady_clock::now();
    cout << "Generate code table time used: " << chrono::duration_cast<chrono::milliseconds>(end1 - start).count() << " ms" << endl;
    cv::Mat img;
    cout << "raw len:" << WIDTH * HEIGHT << endl;
    for (int i = 1; i <= 1; i++)
    {
        auto end1 = chrono::steady_clock::now();
        cout << "|>" << i << ":" << endl;
        img = cv::imread("../img/pic" + to_string(i) + ".jpg", 0);
        cv::threshold(img, img, 200, 255, cv::THRESH_BINARY);
        auto rawBits = mat2Bits(img);
        auto encodeRes = encode(img);
        auto decodeRes = decode(encodeRes);
        cout << "encode len: " << encodeRes.size() << endl;
        cout << "decode len: " << decodeRes.size() << endl;
        cout << "compress ratio: " << (double)encodeRes.size() / (double)rawBits.size() << endl;
        cout << "decode compare to raw result: " << (std::equal(rawBits.begin(), rawBits.end(), decodeRes.begin()) ? "same" : "different") << endl;
        // cout << encodeIdx.size() << endl;
        // cout << decodeIdx.size() << endl;
        // cout << encodeIdx.compare(decodeIdx) << endl;
        auto end2 = chrono::steady_clock::now();
        cout << "encode & decode used: " << chrono::duration_cast<chrono::milliseconds>(end2 - end1).count() << " ms" << endl;
        cout << endl;
    }
    return 0;
}
