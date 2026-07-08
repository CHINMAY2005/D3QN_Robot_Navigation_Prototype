import puppeteer from 'puppeteer';

(async () => {
  const browser = await puppeteer.launch({
    headless: "new",
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 800 });
  await page.goto('http://localhost:5173', { waitUntil: 'networkidle2' });
  
  // Wait a moment for rendering
  await new Promise(r => setTimeout(r, 2000));
  
  await page.screenshot({ path: '../artifacts/ui_screenshot.png' });
  
  await browser.close();
})();
