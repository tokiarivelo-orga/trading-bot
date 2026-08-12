"use server";

import { spawn } from "child_process";
import path from "path";

export async function triggerManualTraining(startDate?: string, endDate?: string, symbol?: string) {
  const rootDir = path.join(process.cwd(), "..");
  
  try {
    const env = { ...process.env };
    if (startDate) env.START_DATE = startDate;
    if (endDate) env.END_DATE = endDate;
    if (symbol) env.SYMBOL = symbol;

    // Spawn training process in the background so it doesn't block the request
    const child = spawn("make", ["train-dl"], {
      cwd: rootDir,
      env: env,
      detached: true, // Run independently of the Node process
      stdio: "ignore", // Don't hold on to stdio pipes
    });
    
    // Unref allows the parent process to exit independently of the child
    child.unref();
    
    return { success: true, message: "Entraînement manuel lancé en arrière-plan !" };
  } catch (error) {
    console.error("Failed to start training:", error);
    return { success: false, message: "Erreur lors du lancement de l'entraînement." };
  }
}

export async function toggleBotState(symbol: string, activate: boolean) {
  const modelName = symbol === "XAUUSD" ? "smc_dl_m5" : "smc_dl_m5_step200";
  const url = `http://localhost:8000/skills/normal/${encodeURIComponent(symbol)}/bots`;
  
  try {
    if (activate) {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Session-Token": "dev-session-token" },
        body: JSON.stringify({ strategy_name: modelName, bot_name: modelName, risk_multiplier: 1.0 })
      });
      if (!res.ok) throw new Error(await res.text());
      return { success: true, message: `Bot activé pour ${symbol}` };
    } else {
      const res = await fetch(`${url}/${modelName}`, {
        method: "DELETE",
        headers: { "X-Session-Token": "dev-session-token" },
      });
      if (!res.ok) throw new Error(await res.text());
      return { success: true, message: `Bot désactivé pour ${symbol}` };
    }
  } catch (err: any) {
    return { success: false, message: `Erreur: ${err.message}` };
  }
}

export async function checkBotState(symbol: string) {
  const modelName = symbol === "XAUUSD" ? "smc_dl_m5" : "smc_dl_m5_step200";
  try {
    const res = await fetch("http://localhost:8000/skills/normal", {
        headers: { "X-Session-Token": "dev-session-token" },
        cache: 'no-store'
    });
    const skills = await res.json();
    const isActive = skills.some((s: any) => s.symbol === symbol && s.bot_name === modelName);
    return { active: isActive };
  } catch (err) {
    return { active: false };
  }
}
