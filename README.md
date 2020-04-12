# NTUST Captcha 

對應選課系統及校務(學生)系統



2份辨識基本上是一份檔案，只差在crop不同，還有需要resize

後續使用裝置是行動平台，附上tflite

版本為`tensorflow 2.1.0`

## 選課系統



![](https://github.com/takidog/CNN_NTUST_Captcha/blob/master/course_selection_simle.png?raw=true)

選課系統Captcha圖片

simple 300~500張並沒有差別太多`accuracy 0.88`

### 預處理過程

`crop((5, 3, 66, 17))` 裁切61x14



## 校務系統

![](https://github.com/takidog/CNN_NTUST_Captcha/blob/master/std_simple.png?raw=true)

校務系統Captcha圖片

simple 200張`accuracy 0.99`

可直接OCR，基本維持9成

### 預處理過程

`crop((10, 10, 124, 36))` 裁切114x26

`resize((57,13)) ` resize一半





---

這份code沒有爬蟲的部分

校務系統可以配合OCR來製作dataset

選課系統PIL簡單處理後OCR後也能到6~7成

給要做生成器的朋友 

圖片參數: 

​    font:`arial bold italic`

​    font size: 15pt

可以嘗試使用這範圍的文字漸層

(25, 0, 200)～(100, 10, 70)



參考:

[https://notes.andywu.tw/2019/%e7%94%a8tensorflowkeras%e8%a8%93%e7%b7%b4%e8%be%a8%e8%ad%98%e9%a9%97%e8%ad%89%e7%a2%bc%e7%9a%84cnn%e6%a8%a1%e5%9e%8b/](https://notes.andywu.tw/2019/用tensorflowkeras訓練辨識驗證碼的cnn模型/)