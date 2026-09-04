const fs = require('fs');
const path = require('path');

function walk(dir, callback) {
  fs.readdirSync(dir).forEach(file => {
    let fullPath = path.join(dir, file);
    if (fs.statSync(fullPath).isDirectory()) {
      walk(fullPath, callback);
    } else if (fullPath.endsWith('.ts') || fullPath.endsWith('.tsx')) {
      callback(fullPath);
    }
  });
}

walk('frontend/src', file => {
  let content = fs.readFileSync(file, 'utf8');
  let original = content;

  // Match .toISOString().replace('T', ' ').slice(5, 16) -> MM-DD HH:MM
  content = content.replace(/\.toISOString\(\)\s*\.\s*replace\(\s*['"]T['"]\s*,\s*['"] ['"]\s*\)\s*\.\s*slice\(\s*5\s*,\s*16\s*\)/g, 
    ".toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })");

  // Match .toISOString().replace('T', ' ').slice(11, 19) -> HH:MM:SS
  content = content.replace(/\.toISOString\(\)\s*\.\s*replace\(\s*['"]T['"]\s*,\s*['"] ['"]\s*\)\s*\.\s*slice\(\s*11\s*,\s*19\s*\)/g, 
    ".toLocaleString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' })");

  if (content !== original) {
    console.log(`Updated ${file}`);
    fs.writeFileSync(file, content, 'utf8');
  }
});
