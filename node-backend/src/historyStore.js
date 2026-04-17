const fs = require("fs/promises");
const path = require("path");

async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

async function readJsonIfExists(filePath, fallback) {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return JSON.parse(raw);
  } catch (err) {
    if (err && (err.code === "ENOENT" || err.code === "ENOTDIR")) return fallback;
    throw err;
  }
}

async function writeJsonAtomic(filePath, data) {
  const dir = path.dirname(filePath);
  await ensureDir(dir);
  const tmpPath = `${filePath}.tmp`;
  await fs.writeFile(tmpPath, JSON.stringify(data, null, 2), "utf8");
  await fs.rename(tmpPath, filePath);
}

function createHistoryStore({ dataDir }) {
  const historyPath = path.join(dataDir, "history.json");

  return {
    async list() {
      const items = await readJsonIfExists(historyPath, []);
      return Array.isArray(items) ? items : [];
    },
    async get(id) {
      const items = await this.list();
      return items.find((x) => x.id === id) || null;
    },
    async add(entry) {
      const items = await this.list();
      const next = [entry, ...items].slice(0, 200);
      await writeJsonAtomic(historyPath, next);
      return entry;
    },
    async clear() {
      await writeJsonAtomic(historyPath, []);
    },
  };
}

module.exports = { createHistoryStore };

