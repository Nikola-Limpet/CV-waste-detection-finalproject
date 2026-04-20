# Smart Waste Vision — Presentation Script (4 Members)

**Total Time:** 18 minutes + Q&A  
**Slides:** 17 slides

---

## Role Assignment

| Member | Section | Slides | Time |
|--------|---------|--------|------|
| **Member A** | Introduction + Objectives | Slides 1-3 | ~3.5 min |
| **Member B** | Literature Review + Dataset | Slides 4-7 | ~5 min |
| **Member C** | Methodology + Results | Slides 8-13 | ~6 min |
| **Member D** | Demo + Conclusion + Q&A | Slides 14-17 | ~3.5 min + Q&A |

---

## MEMBER A — Introduction & Objectives (3.5 min)

### Slide 1: Title Slide (30 seconds)

> Good morning/afternoon everyone, and thank you for joining us.
>
> Our project is called **Smart Waste Vision** — an AI-based waste detection and classification system using Convolutional Neural Networks.
>
> This is our Computer Vision final project. Today, our team will walk you through how we built, trained, and deployed a waste image classifier that can automatically sort waste into 10 categories.

**[Click to Slide 2]**

### Slide 2: The Problem (1.5 minutes)

> Let me start with the problem we are trying to solve.
>
> In Cambodia, waste management is one of the most serious environmental challenges. Cities like Phnom Penh generate **thousands of tons of municipal waste every single day**. But the reality is — most of this waste goes to landfills **completely unsorted**.
>
> Why does this matter? Three reasons:
>
> First, **environmental pollution** — unsorted waste contaminates soil, water, and air. Recyclable materials that could be recovered are lost.
>
> Second, **manual sorting fails** — it is labor-intensive, inconsistent, and inaccurate. Workers cannot sort fast enough or accurately enough.
>
> Third, **health hazards** — untreated organic waste creates disease vectors in urban areas.
>
> So how can Computer Vision help? On the right side you can see our solution:
>
> CNNs combined with transfer learning can **automate waste sorting from photographs**. This directly applies the techniques we learned in our Computer Vision course to a real-world environmental problem.
>
> Our **target** was to build an image classifier that achieves **85% or higher accuracy** on waste categorization. And I can already tell you — we exceeded that significantly. But more on that later.

**[Click to Slide 3]**

### Slide 3: Project Objectives (1.5 minutes)

> Here are our five project objectives:
>
> **Objective 1:** Develop a CNN image classifier that can classify waste into **10 different categories** — including things like plastic, glass, metal, paper, and biological waste.
>
> **Objective 2:** We did not just train one model. We compared **4 different transfer learning architectures** — EfficientNet-B3, ResNet50, VGG16, and MobileNetV2 — to understand which architecture performs best for waste classification.
>
> **Objective 3:** Apply proper Computer Vision preprocessing techniques — image resizing, normalization, data augmentation, and feature extraction.
>
> **Objective 4:** Evaluate our models using standard metrics like accuracy and confusion matrices, plus we used **Grad-CAM** to visualize what the model actually "sees" when it makes a prediction. This provides interpretability — we can explain WHY the model makes each decision.
>
> **Objective 5:** Build a functioning **web demo** so users can upload a waste image and get an instant classification with disposal guidance.
>
> Our scope is focused on **image classification only** — not object detection or segmentation. Hardware integration is outside our scope but noted as future work.
>
> Now I will hand over to [Member B] who will cover our literature review and dataset.

---

## MEMBER B — Literature Review & Dataset (5 min)

### Slide 4: Literature Review — CNN Architectures (1.5 minutes)

> Thank you, [Member A].
>
> Let me explain the research foundation behind our model choices.
>
> We selected 4 architectures, each with a different design philosophy:
>
> **ResNet** by He et al. in 2016 introduced **residual connections** — skip connections that solve the vanishing gradient problem. This is what allows us to train networks with 50 or more layers without the gradients disappearing.
>
> **VGG16** by Simonyan and Zisserman in 2015 uses a simpler, sequential design with small 3x3 convolutions stacked together. We use it as our **baseline reference** — a well-known architecture to compare against.
>
> **MobileNetV2** by Sandler et al. in 2018 uses **depthwise separable convolutions** and inverted residual blocks. This makes it extremely lightweight — only 2.6 million parameters. This is relevant for deployment in **resource-constrained environments** like mobile devices in Cambodia.
>
> **EfficientNet** by Tan and Le in 2019 introduced **compound scaling** — the idea of scaling depth, width, and resolution together instead of separately. This was actually added later in our project after the first three models plateaued at around 80-83% accuracy. And it made a huge difference.
>
> At the bottom, you can see **Grad-CAM** by Selvaraju et al. — this is how we visualize model attention. It creates heatmaps showing which regions of the image the model focuses on when making a prediction.

**[Click to Slide 5]**

### Slide 5: Transfer Learning & Domain Research (1 minute)

> On the left, we have recent **domain-specific research** that validates our approach:
>
> A 2025 paper in Scientific Reports on intelligent waste sorting, and a 2024 paper in IJACSA on AI-powered waste classification using CNNs — both confirm that this is an active research area and that our CNN approach is well-suited for this task.
>
> On the right — **why does transfer learning work here?** ImageNet does not contain waste images. So why use ImageNet-pretrained models?
>
> The key insight is that **low-level features transfer across domains**. The edges, textures, color patterns, and shapes that a model learns from millions of ImageNet images are useful for recognizing waste too. We freeze the backbone, then fine-tune the higher layers to specialize for waste.
>
> An important decision: we **added EfficientNet-B3** after our initial experiments showed all three original models plateauing at 80-83%. Compound scaling delivered **93.96% accuracy** — a massive improvement.

**[Click to Slide 6]**

### Slide 6: Dataset Overview (1.5 minutes)

> Now let me describe our dataset.
>
> We used the **Garbage Classification V2** dataset from Kaggle. It contains **13,347 RGB images** standardized to 256x256 pixels.
>
> The dataset has **10 waste categories**, which we color-coded by disposal type:
>
> - **Red** for hazardous — that is Battery
> - **Green** for compostable — Biological waste
> - **Blue** for recyclable — Cardboard, Clothes, Glass, Metal, Paper, and Plastic
> - **Gray** for landfill — Shoes and Trash
>
> For our data split, we used a **stratified 70/15/15 split** — 9,343 images for training, 2,002 for validation, and 2,003 for testing. The stratification ensures that each class is proportionally represented in all three sets.
>
> We used a fixed **seed of 42** so our results are fully reproducible.
>
> To handle **class imbalance**, we computed balanced class weights using scikit-learn and passed them to the training process. This makes the model penalize misclassifications of minority classes more heavily.

**[Click to Slide 7]**

### Slide 7: Data Preprocessing Pipeline (1 minute)

> Here is our preprocessing pipeline.
>
> **Step 1:** Resize all images to **224x224** — the standard input size for pretrained CNN architectures.
>
> **Step 2:** **Normalize** pixel values to the [0, 1] range.
>
> **Step 3:** Apply a custom **BackbonePreprocess** layer — this is a Keras layer we wrote ourselves that applies model-specific ImageNet standardization. It is registered as a serializable custom layer, which avoids Lambda issues and makes our models fully self-contained for deployment.
>
> **Step 4:** We use a **tf.data pipeline** with cache, shuffle, batch size 32, and prefetch for optimal GPU utilization.
>
> For **data augmentation** — applied only during training — we use horizontal flip, rotation of plus or minus 20 degrees, zoom, brightness, and contrast adjustments. Validation and test data remain unaugmented.
>
> Now [Member C] will explain our methodology and results.

---

## MEMBER C — Methodology & Results (6 min)

### Slide 8: Transfer Learning Architecture (1.5 minutes)

> Thank you, [Member B].
>
> Let me walk you through how our models are built and trained.
>
> At the top, you can see our **unified architecture** that all 4 models share. The flow goes:
>
> **Input** image at 224x224 → **Rescale** by 255 → pass through **BackbonePreprocess** for model-specific normalization → then into the **Frozen Backbone** which is pretrained on ImageNet → **Global Average Pooling** to reduce spatial dimensions → a **Dense layer with 256 neurons and ReLU** → **Dropout at 0.5** for regularization → and finally a **Dense 10 with Softmax** output for our 10 waste categories.
>
> In the table below, you can see how the 4 models differ:
>
> **EfficientNet-B3** has 11.2 million parameters with compound scaling — we unfreeze 40 layers during fine-tuning.
>
> **ResNet50** has 24.1 million parameters with residual connections — 30 layers unfrozen.
>
> **VGG16** has 14.8 million parameters — but we only unfreeze 8 layers because it has fewer meaningful blocks.
>
> **MobileNetV2** is the lightest at only 2.6 million parameters — 30 layers unfrozen.

**[Click to Slide 9]**

### Slide 9: Two-Phase Training (1.5 minutes)

> Our training follows a **two-phase strategy**:
>
> **Phase 1 — Head Training** for 10 epochs: We freeze the entire backbone so the pretrained ImageNet weights cannot change. We only train the classification head — the Global Average Pooling and Dense layers — at a learning rate of **1e-3**. The purpose is to teach the model how to map backbone features to our 10 waste categories.
>
> **Phase 2 — Fine-Tuning** for 20 more epochs: We unfreeze the last N layers of the backbone and reduce the learning rate to **1e-5**. This much lower learning rate prevents **catastrophic forgetting** — we do not want to destroy the useful ImageNet features.
>
> There is one **critical detail** here: we keep all **BatchNormalization layers frozen** even during Phase 2. Why? Because with our small batch sizes of 32 or 16, the batch statistics would be very noisy. Keeping BN frozen preserves the stable running mean and variance from ImageNet.
>
> On the right, you can see our training configuration: Adam optimizer, sparse categorical cross-entropy loss, early stopping with patience 10, learning rate reduction on plateau, and model checkpointing.
>
> One practical note — VGG16 required **batch size 16** instead of 32 because of the 6GB VRAM constraint on our RTX 4050 GPU. We also trained each model in **separate subprocesses** to fully release GPU memory between runs.

**[Click to Slide 10]**

### Slide 10: Grad-CAM (30 seconds)

> For **model interpretability**, we implemented Grad-CAM.
>
> The process works in 6 steps: forward pass to the last convolutional layer, compute gradients using TensorFlow's GradientTape, global average pool the gradients to get importance weights, compute a weighted sum of feature maps, apply ReLU and normalize, then overlay as a color heatmap on the original image.
>
> This tells us **where the model looks** when making each prediction — essential for building trust in an automated waste sorting system.

**[Click to Slide 11]**

### Slide 11: Results — Model Performance (1 minute)

> Now for our results — the part everyone has been waiting for.
>
> Our **target was 85% accuracy**. As you can see, we **achieved 93.96%** with EfficientNet-B3 — exceeding the target by nearly **9 percentage points**.
>
> Looking at the bar chart:
>
> **EfficientNet-B3** leads clearly at **93.96%** with 11.2 million parameters.
>
> **ResNet50** achieved **82.93%** — solid, but significantly lower despite having 24.1 million parameters — more than double.
>
> **MobileNetV2** achieved **81.73%** with only 2.6 million parameters — the most lightweight model.
>
> **VGG16** came last at **80.33%** despite having 14.8 million parameters.
>
> All four models achieve inference under 100 milliseconds per image, making **real-time classification feasible**.

**[Click to Slide 12]**

### Slide 12: Key Findings (1 minute)

> Here are our key analysis findings:
>
> First, **EfficientNet-B3 dominates** — the 11% accuracy gap over ResNet50 shows that compound scaling — scaling depth, width, and resolution together — is clearly superior to just using a deeper network.
>
> Second, **more parameters does not mean better accuracy**. VGG16 has 14.8 million parameters but performs worse than MobileNetV2 which has only 2.6 million. Architecture design matters more than raw parameter count.
>
> Third, in terms of **efficiency**, MobileNetV2 is the winner at 31.4% accuracy per million parameters — making it the best choice for mobile or edge deployment.
>
> And regarding **confusion patterns** — the model struggles most with visually similar categories: Clothes versus Shoes, Paper versus Cardboard, and general Trash versus specific categories. This makes intuitive sense.

**[Click to Slide 13]**

### Slide 13: Grad-CAM Visualizations (30 seconds)

> Finally, our Grad-CAM analysis confirms that the model learns meaningful features.
>
> On **correct predictions**, the model focuses on discriminative object features — bottle shape for glass, woven texture for clothes, metallic shine for metal.
>
> On **incorrect predictions**, the attention drifts to background textures or ambiguous regions — this explains why visually similar classes cause confusion.
>
> This Grad-CAM analysis is important because it shows our model is not just memorizing patterns — it is actually learning to recognize waste features.
>
> Now [Member D] will show you the live demo and our conclusions.

---

## MEMBER D — Demo & Conclusion (3.5 min + Q&A)

### Slide 14: Demo Features (1 minute)

> Thank you, [Member C].
>
> We did not just train models — we built a full **production-quality web application** using Gradio. Over 900 lines of code with 6 feature-rich tabs:
>
> **Tab 1 — Classify:** Users can upload an image or use their webcam. The app shows top-5 confidence scores, a Grad-CAM heatmap overlay, disposal guidance with specific steps, and even AI-powered advice from a Gemini language model.
>
> **Tab 2 — Live Webcam:** Real-time detection with auto-detect every 2 seconds.
>
> **Tab 3 — Batch Processing:** Upload multiple images at once and export results as CSV.
>
> **Tab 4 — AI Chat:** An EcoBot assistant powered by Gemini that can answer waste disposal and recycling questions.
>
> **Tab 5 — History & Stats:** A session dashboard with pie charts showing classification distribution and environmental impact metrics.
>
> **Tab 6 — Waste Guide:** A searchable educational reference for all 10 categories with decomposition times and safety alerts.
>
> We also have **model switching** so you can compare models live, and **confidence warnings** for low-confidence predictions.

**[Click to Slide 15]**

### Slide 15: Live Demo (2 minutes)

> Now let me **show you the application live**.
>
> *[Open the Gradio app at localhost:7860]*
>
> *[Follow this flow — adjust based on what works:]*
>
> First, here is the interface. You can see all 6 tabs at the top.
>
> Let me upload a waste image... *[upload an image]* ... and you can see the classification result with the confidence score, the Grad-CAM heatmap showing where the model focuses, and the disposal guidance telling you which bin to use and how to dispose of it properly.
>
> Notice the **color-coded bin system**: blue for recyclable, green for compostable, red for hazardous, and gray for landfill.
>
> Let me also quickly show the **batch processing** — I can upload several images and get a summary table with CSV export.
>
> And if I switch to the **Waste Guide** tab, you can see educational information for each category.
>
> *[If time permits, show a quick EcoBot chat]*

**[Click to Slide 16]**

### Slide 16: Conclusion & Future Work (1 minute)

> To wrap up our presentation:
>
> Our **key achievement** — 93.96% test accuracy with EfficientNet-B3, exceeding our 85% target by nearly 9 points.
>
> We successfully compared **4 transfer learning architectures** and demonstrated that architecture design matters more than parameter count.
>
> We provided **Grad-CAM interpretability** — showing not just what the model predicts but why.
>
> And we delivered a **6-tab production-quality web demo** with AI integration, webcam support, and batch processing.
>
> We are also **honest about our limitations**: this is single-object classification only — not detection. The dataset uses Western waste images, not Cambodian-specific. And clean isolated objects differ from real-world mixed conditions.
>
> For **future work**, we propose: YOLO for multi-object detection, TensorFlow Lite for mobile deployment, collecting a Cambodian-specific waste dataset, real-time video processing, and ensemble methods combining multiple models.

**[Click to Slide 17]**

### Slide 17: Thank You & Q&A

> Thank you for your attention. We are now happy to take any questions.

---

## Q&A — ALL MEMBERS (Assigned by Topic)

**Who answers which questions:**

| Topic | Who Answers |
|-------|------------|
| Problem motivation, project scope | Member A |
| Literature, dataset, preprocessing | Member B |
| Architecture, training, results, Grad-CAM | Member C |
| Demo, deployment, future work | Member D |

### Prepared Answers

**Q: "Why did you add EfficientNet-B3 when it was not in the original proposal?"**
*(Member C answers)*

> After our initial experiments, ResNet50, VGG16, and MobileNetV2 all plateaued at 80 to 83% accuracy. We researched EfficientNet because its compound scaling approach was designed to achieve better accuracy with fewer parameters. Our results validated this decision — 93.96% with only 11.2 million parameters versus ResNet50's 82.93% with 24.1 million.

**Q: "Why is there no custom CNN baseline?"**
*(Member C answers)*

> We prioritized adding a fourth transfer learning architecture over a custom baseline. Comparing four models with fundamentally different design philosophies — residual connections, sequential convolutions, depthwise separable, and compound scaling — provides more insightful comparisons than including a custom CNN that would obviously underperform all pretrained models.

**Q: "How do you handle class imbalance?"**
*(Member B answers)*

> We compute balanced class weights using scikit-learn's compute_class_weight function and pass them directly to model.fit. This adjusts the loss function to penalize misclassifications of minority classes more heavily, without requiring oversampling which could lead to overfitting.

**Q: "Why keep BatchNormalization frozen during fine-tuning?"**
*(Member C answers)*

> With batch sizes of 32 or 16, the per-batch statistics would be very noisy and not representative of the true data distribution. Keeping BatchNorm layers frozen preserves the stable running mean and variance that were learned from millions of ImageNet images. This is critical for training stability — unfreezing BN with small batches often causes training to diverge.

**Q: "What are the main failure modes?"**
*(Member C answers)*

> The primary failures are confusion between visually similar categories. Clothes and Shoes both have fabric or leather textures. Paper and Cardboard have similar color and texture. And general Trash can look like many specific categories. Our Grad-CAM analysis on misclassified samples shows the model sometimes focuses on background textures rather than the actual object.

**Q: "Could this work in real-world Cambodia?"**
*(Member A or D answers)*

> The foundation is solid, but real-world deployment needs several things: First, a Cambodian-specific dataset with local waste items — our current dataset uses Western waste images. Second, object detection using YOLO or Faster R-CNN for images with multiple waste items. Third, TensorFlow Lite optimization for mobile deployment — MobileNetV2 is already lightweight at 2.6 million parameters. Fourth, handling dirty, contaminated, or partially occluded items. The architecture and training pipeline we built would transfer directly to a local dataset.

**Q: "Why Gradio over Streamlit?"**
*(Member D answers)*

> Gradio has built-in support for image upload, webcam capture, and machine learning model serving with minimal code. Its Blocks API lets us create complex multi-tab layouts — we built 6 tabs. It is also deployable on Hugging Face Spaces and supports Docker containers, which we use for our Railway deployment.

**Q: "Explain Grad-CAM technically."**
*(Member C answers)*

> First, we do a forward pass to get the last convolutional layer's feature maps. Then we compute gradients of the predicted class score with respect to those feature maps using TensorFlow's GradientTape. We global average pool the gradients to get one importance weight per channel. Then we compute a weighted combination of the feature maps, apply ReLU to keep only positive influences, normalize to 0 to 1, resize to input dimensions, and overlay as a colored heatmap. The gradients tell us how important each feature map channel is, and the feature maps tell us where in the image those features are activated.

**Q: "What would you do differently if you started over?"**
*(Any member answers)*

> We would include a formal hyperparameter search — grid search or Bayesian optimization — instead of manual tuning. We would also try to collect at least a small Cambodian waste dataset for domain adaptation testing. We would feature per-class precision and recall more prominently. And we would experiment with ensemble methods — combining predictions from multiple models.

**Q: "Why are inference times different across models?"**
*(Member C answers)*

> Inference time depends on computational complexity, not just parameter count. EfficientNet-B3 uses compound scaling which adds computational depth, so it takes 83 milliseconds. VGG16 and ResNet50 run at about 58 milliseconds because their architectures are highly optimized on GPU. MobileNetV2, while having the fewest parameters, uses depthwise separable convolutions that involve more memory-bound operations, resulting in 79 milliseconds.

---

## Presentation Tips

### Before the Presentation
- [ ] Test the Gradio app works: `python app/app.py`
- [ ] Prepare 3-4 test images (clear, recognizable waste items) for the live demo
- [ ] Have a backup plan if webcam or internet fails (pre-saved screenshots)
- [ ] Each member practices their section at least 2 times
- [ ] Time each section to stay within the allocated time

### During the Presentation
- **Transitions:** When handing off, say the next person's name: "Now I will hand over to [Name] who will cover..."
- **Eye contact:** Look at the audience, not the slides
- **Pointing:** When referring to charts or diagrams, point to the specific element
- **Pace:** Speak slowly and clearly — nervousness makes people speak too fast
- **Slide 11 reveal:** Build suspense by showing VGG16 and MobileNetV2 results first, then ResNet50, then reveal the 93.96%

### For the Demo (Member D)
- Have the app **already running** before the presentation starts
- Keep a browser tab open and ready to switch
- If something fails during the demo, say: "As you can see from the screenshots on the previous slide..." and move on
- Do NOT try to debug live — just switch to the backup plan

### For Q&A (All Members)
- If you do not know the answer, say: "That is a great question. Based on our understanding..." and give your best answer
- If the question is for another member's area, say: "[Name] worked on that part — [Name], would you like to answer?"
- Keep answers under 30 seconds — be concise
