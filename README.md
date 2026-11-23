# kids-yt-bot — دليل الإعداد

فيديو كرتوني 3D للأطفال (دقيقة) يُنتَج ويُنشَر تلقائياً 3 مرات يومياً عبر GitHub Actions.

## الخطوة 1: مفاتيح الخدمات
1. **GEMINI_API_KEY**: من Google AI Studio ← Get API key.
2. **HF_TOKEN** (اختياري للموسيقى): huggingface.co/settings/tokens ← نوع Read.
3. **يوتيوب**: في Google Cloud Console أنشئ مشروعاً ← فعّل YouTube Data API v3 ← OAuth client من نوع Desktop ← نزّل `client_secret.json`.
   ثم على حاسوبك: `pip install google-auth-oauthlib` ثم `python get_refresh_token.py client_secret.json`.
   سجّل الدخول بحساب القناة، وانسخ القيم الثلاث المطبوعة.

## الخطوة 2: GitHub
1. أنشئ مستودعاً (يُفضَّل **عاماً** لدقائق Actions غير المحدودة، ولا تضع أي مفتاح في الكود).
2. ارفع كل الملفات (مع مجلد `.github`).
3. Settings ← Secrets and variables ← Actions ← أضف:
   `GEMINI_API_KEY`, `HF_TOKEN`, `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN`.
4. (Variables) أضف `PRIVACY` = `private` للاختبار، ثم غيّرها إلى `public` لاحقاً.

## الخطوة 3: التجربة
Actions ← kids-video ← Run workflow (dry_run = 1). حمّل `final.mp4` من Artifacts وشاهده.
بعدها شغّله مع dry_run = 0 ليرفع فيديو **خاصاً** إلى قناتك.

## ملاحظات
- المواعيد بتوقيت UTC في `daily.yml` (6/11/16 = 8/13/18 بتوقيت ليبيا).
- إن لم تكفِ حصة صور Gemini المجانية، يتحول السكربت تلقائياً إلى FLUX على Hugging Face.
- مشاريع YouTube API غير المعتمدة تُجبر الفيديوهات على الوضع الخاص حتى تطلب تدقيق API من Google.
- الفيديو يُوسَم «موجّه للأطفال» ومحتوى مولَّد بالذكاء الاصطناعي. راجع سياسات يوتيوب للمحتوى المتكرر.
- أسماء النماذج في أعلى `main.py` (TEXT_MODEL, IMAGE_MODEL) قد تتغير؛ حدّثها حسب AI Studio.
