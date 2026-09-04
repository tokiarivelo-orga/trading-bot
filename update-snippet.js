const fs = require('fs');

function update(filePath, relativePathToChartFormat) {
  let content = fs.readFileSync(filePath, 'utf8');

  // Add import if missing
  if (!content.includes('getLocalTimeZoneOptions')) {
    const importStmt = `import { getLocalTimeZoneOptions } from "${relativePathToChartFormat}";\n`;
    const lastImportIndex = content.lastIndexOf('import ');
    const newlineAfterImport = content.indexOf('\n', lastImportIndex);
    content = content.slice(0, newlineAfterImport + 1) + importStmt + content.slice(newlineAfterImport + 1);
  }

  // Update createChart configuration
  const regex = /const chart = createChart\(container, \{\s*(.*?)\s*\}\);/s;
  const match = content.match(regex);
  if (match) {
    let optsStr = match[1];
    
    if (!optsStr.includes('timeZoneOpts')) {
       // Insert timeZoneOpts declaration before createChart
       const timeZoneOptsStr = `const timeZoneOpts = getLocalTimeZoneOptions();\n    const chart = createChart(container, {`;
       content = content.replace(/const chart = createChart\(container, \{/, timeZoneOptsStr);
       
       // Add to timeScale and localization
       content = content.replace(/timeScale: \{([^}]+)\},/, 'timeScale: {$1, tickMarkFormatter: timeZoneOpts.timeScale.tickMarkFormatter },\n      localization: timeZoneOpts.localization,');
    }
  }

  fs.writeFileSync(filePath, content, 'utf8');
}

update('frontend/src/shared/ui/DecisionChartSnippet.tsx', '@/features/chart/chartFormat');

