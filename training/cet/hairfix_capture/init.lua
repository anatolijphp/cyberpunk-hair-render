-- Training captures for the hair fix (see src/capture.h). Development tool, needs Cyber Engine
-- Tweaks. Install: copy this folder to bin/x64/plugins/cyber_engine_tweaks/mods/hairfix_capture/.
--
-- The mod's capture key (hairfix.ini [Debug] CaptureKey) writes pair.txt here; this script asks the
-- mod for captures through request.txt and waits for status.txt.
-- - outside photo mode: one capture in the current FSR quality
-- - in photo mode: composition COMPOSITION, then one capture per character rotation in ROTATIONS (photo mode attribute 7,
--   relative to the current rotation), then the rotation is restored. Do the same in the other FSR
--   quality (leave photo mode, switch the quality, enter photo mode again without moving) to get
--   pairs: the same attribute values give the same views.
-- The FSR quality cannot be switched from here: the game applies it only from its settings menu.
-- Progress goes to progress.txt (CET's own log reaches the disk late).

local ROTATIONS = { 0, 45, 90, 135, 180, 225, 270, 315 }
local ATTR_ROTATION = 7          -- photo mode: character rotation in degrees
local ATTR_COMPOSITION = 23      -- photo mode: camera composition preset (sets field of view and framing)
local COMPOSITION = 3
local POSE_SETTLE = 10           -- seconds after a rotation before capturing: the hair swings for a while
local COMPOSE_SETTLE = 1.5       -- seconds after the composition changed (camera only)
local CAPTURE_TIMEOUT = 120      -- seconds; the capture runs until every jitter was seen in every dither phase

local state = "idle"
local lastPair = nil
local setId, poses, pose, timer, requestId, baseRotation = nil, {}, 0, 0, nil, 0

local photoMenu = nil            -- photo mode menu controller (set by any attribute change)
local attrValues = {}            -- last value of each photo mode attribute

local function readFile(name)
    local f = io.open(name, "r")
    if not f then return nil end
    local s = f:read("*a")
    f:close()
    return s
end

local function writeFile(name, text)
    local f = io.open(name, "w")
    if not f then return false end
    f:write(text)
    f:close()
    return true
end

local function say(message)
    print("[hairfix] " .. message)
    local f = io.open("progress.txt", "a")
    if f then
        f:write(os.date("%H:%M:%S ") .. message .. "\n")
        f:close()
    end
end

local function renderSize()
    local s = readFile("render.txt")
    return s and s:match("%d+ %d+") or "?"
end

local function photoModeActive()
    local ok, r = pcall(function() return Game.GetPhotoModeSystem():IsPhotoModeActive() end)
    return ok and r
end

local function setAttribute(key, value)
    local ok, err = pcall(function() photoMenu:OnAttributeUpdated(key, value, true) end)
    if not ok then say("setting photo mode attribute " .. key .. " failed: " .. tostring(err)) end
    return ok
end

-- one capture by the mod
local function requestCapture(tag)
    requestId = tostring(setId) .. string.format("%02d", pose)
    writeFile("request.txt", requestId .. " " .. tag .. "\n")
    timer = CAPTURE_TIMEOUT
    state = "capture"
end

local function finish(message)
    if state ~= "idle" and #poses > 0 and photoMenu then setAttribute(ATTR_ROTATION, baseRotation) end
    say(message)
    state = "idle"
end

registerForEvent("onInit", function()
    lastPair = readFile("pair.txt")   -- a request left over from an earlier session is not served
    local ok, err = pcall(function()
        Observe("gameuiPhotoModeMenuController", "OnAttributeUpdated", function(self, key, value)
            photoMenu = self
            attrValues[key] = value
        end)
    end)
    if not ok then say("photo mode hook failed: " .. tostring(err)) end
    say("capture script ready")
end)

registerForEvent("onUpdate", function(dt)
    if state == "idle" then
        local p = readFile("pair.txt")
        if not p or p == lastPair then return end
        lastPair = p
        setId = p:match("%d+") or tostring(os.time())
        pose = 0
        if not photoModeActive() then
            poses = {}
            say("single capture, render " .. renderSize())
            pose = 1
            requestCapture("c" .. setId .. "_w" .. renderSize():match("%d+"))
            return
        end
        if not photoMenu then
            say("touch any photo mode slider once, then press the capture key again")
            return
        end
        baseRotation = attrValues[ATTR_ROTATION] or 0
        poses = ROTATIONS
        say(string.format("photo mode set %s: composition %d, %d rotations from %s degrees, render %s", setId,
            COMPOSITION, #poses, tostring(baseRotation), renderSize()))
        if not setAttribute(ATTR_COMPOSITION, COMPOSITION) then
            finish("cannot set the composition")
            return
        end
        timer = COMPOSE_SETTLE
        state = "compose"
    elseif state == "compose" then
        timer = timer - dt
        if timer <= 0 then state = "next" end
    elseif state == "next" then
        pose = pose + 1
        if pose > #poses then
            finish("set " .. setId .. " done, rotation restored")
            return
        end
        if not setAttribute(ATTR_ROTATION, (baseRotation + poses[pose]) % 360) then
            finish("cannot rotate the character")
            return
        end
        timer = POSE_SETTLE
        state = "settle"
    elseif state == "settle" then
        timer = timer - dt
        if timer <= 0 then
            requestCapture(string.format("pm%s_r%03d_w%s", setId, poses[pose], renderSize():match("%d+")))
        end
    elseif state == "capture" then
        timer = timer - dt
        local s = readFile("status.txt")
        if s and s:match("^" .. requestId .. " ") then
            if not s:match("^%d+ done") then
                finish("capture failed (see hairfix.log)")
            elseif #poses == 0 then
                finish("captured at render " .. renderSize())
            else
                say(string.format("rotation +%d captured", poses[pose]))
                state = "next"
            end
        elseif timer <= 0 then
            finish("capture timed out")
        end
    end
end)
