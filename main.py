from ursina import *
import math
import random
import hashlib

app = Ursina(title='Universe Simulator', borderless=False)

Sky(color=color.black)

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION - OPTIMIZED
# ══════════════════════════════════════════════════════════════════

class Config:
    CHUNK_SIZE = 4000
    RENDER_DISTANCE = 2
    STARS_PER_CHUNK = 2
    PLANET_LOAD_DIST = 800
    PLANET_UNLOAD_DIST = 1200

# ══════════════════════════════════════════════════════════════════
# FLOATING ORIGIN
# ══════════════════════════════════════════════════════════════════

class FloatingOrigin:
    def __init__(self):
        self.world_offset = Vec3(0, 0, 0)
        self.threshold = 2000
        self.entities = []
        
    def register(self, entity):
        if entity not in self.entities:
            self.entities.append(entity)
        
    def unregister(self, entity):
        if entity in self.entities:
            self.entities.remove(entity)
            
    def update(self, player_pos):
        if abs(player_pos.x) > self.threshold or \
           abs(player_pos.y) > self.threshold or \
           abs(player_pos.z) > self.threshold:
            
            shift = Vec3(player_pos.x, player_pos.y, player_pos.z)
            self.world_offset += shift
            
            for entity in self.entities[:]:
                if entity:
                    entity.position -= shift
                    
            return shift
        return Vec3(0, 0, 0)
    
    def get_absolute_position(self, local_pos):
        return self.world_offset + local_pos

floating_origin = FloatingOrigin()

# ══════════════════════════════════════════════════════════════════
# SEEDED RANDOM
# ══════════════════════════════════════════════════════════════════

def get_seed(x, y, z, salt=""):
    data = f"{int(x)},{int(y)},{int(z)},{salt}"
    return int(hashlib.md5(data.encode()).hexdigest()[:8], 16)

# ══════════════════════════════════════════════════════════════════
# STAR TYPES
# ══════════════════════════════════════════════════════════════════

STAR_TYPES = [
    {'color': color.blue, 'glow': color.cyan, 'size': (200, 350), 'prob': 0.03, 'name': 'Blue Giant'},
    {'color': color.cyan, 'glow': color.azure, 'size': (150, 250), 'prob': 0.05, 'name': 'Blue-White'},
    {'color': color.white, 'glow': color.white, 'size': (100, 180), 'prob': 0.10, 'name': 'White Star'},
    {'color': color.yellow, 'glow': color.orange, 'size': (80, 140), 'prob': 0.15, 'name': 'Yellow Dwarf'},
    {'color': color.orange, 'glow': color.yellow, 'size': (60, 110), 'prob': 0.25, 'name': 'Orange Dwarf'},
    {'color': color.red, 'glow': color.orange, 'size': (40, 90), 'prob': 0.42, 'name': 'Red Dwarf'},
]

def get_star_type(rng):
    roll = rng.random()
    cumulative = 0
    for props in STAR_TYPES:
        cumulative += props['prob']
        if roll < cumulative:
            return props
    return STAR_TYPES[-1]

# ══════════════════════════════════════════════════════════════════
# PLANET TYPES
# ══════════════════════════════════════════════════════════════════

PLANET_TYPES = [
    {'color': color.red, 'name': 'Molten', 'size': (8, 15)},
    {'color': color.orange, 'name': 'Desert', 'size': (10, 20)},
    {'color': color.blue, 'name': 'Ocean', 'size': (12, 25)},
    {'color': color.green, 'name': 'Forest', 'size': (12, 22)},
    {'color': color.cyan, 'name': 'Ice', 'size': (10, 20)},
    {'color': color.white, 'name': 'Frozen', 'size': (10, 18)},
    {'color': color.magenta, 'name': 'Exotic', 'size': (8, 18)},
    {'color': color.violet, 'name': 'Crystal', 'size': (8, 16)},
    {'color': color.gray, 'name': 'Rocky', 'size': (6, 14)},
    {'color': color.brown, 'name': 'Barren', 'size': (8, 16)},
    {'color': color.lime, 'name': 'Toxic', 'size': (10, 18)},
    {'color': color.yellow, 'name': 'Sulfur', 'size': (8, 15)},
]

GAS_GIANTS = [
    {'color': color.orange, 'name': 'Gas Giant', 'size': (35, 60)},
    {'color': color.yellow, 'name': 'Yellow Giant', 'size': (40, 65)},
    {'color': color.cyan, 'name': 'Ice Giant', 'size': (30, 50)},
    {'color': color.blue, 'name': 'Blue Giant', 'size': (32, 55)},
]

# ══════════════════════════════════════════════════════════════════
# BACKGROUND STARS (reduced count)
# ══════════════════════════════════════════════════════════════════

background_stars = Entity()
bg_colors = [color.white, color.yellow, color.cyan, color.orange, color.red]

for _ in range(600):  # Reduced from 3000
    theta = random.uniform(0, math.pi * 2)
    phi = random.uniform(-math.pi/2, math.pi/2)
    r = 8000
    
    x = r * math.cos(phi) * math.cos(theta)
    y = r * math.sin(phi)
    z = r * math.cos(phi) * math.sin(theta)
    
    Entity(
        parent=background_stars,
        model='quad',
        position=(x, y, z),
        scale=random.uniform(3, 10),
        color=random.choice(bg_colors),
        billboard=True,
        unlit=True
    )

# ══════════════════════════════════════════════════════════════════
# EFFECTS (simplified)
# ══════════════════════════════════════════════════════════════════

class Effects:
    def __init__(self):
        self.warning = Text(
            text='',
            position=(0, 0.3),
            origin=(0, 0),
            scale=2,
            color=color.red
        )
        self.time_dilation = 1.0
        self.gravity_pull = Vec3(0, 0, 0)
        
    def update(self, ship_pos):
        self.gravity_pull = Vec3(0, 0, 0)
        self.time_dilation = 1.0
        warning = ""
        
        # Only check nearby chunks
        for chunk in universe.chunks.values():
            for obj in chunk.objects:
                dist = (obj.position - ship_pos).length()
                
                if isinstance(obj, BlackHole):
                    grav_r = obj.scale_x * 15
                    if dist < grav_r:
                        strength = (1 - dist / grav_r) ** 2 * 50
                        direction = (obj.position - ship_pos).normalized()
                        self.gravity_pull += direction * strength
                        self.time_dilation = max(0.1, dist / grav_r)
                        
                        if dist < obj.scale_x * 2:
                            warning = "☠ EVENT HORIZON ☠"
                            self.gravity_pull = direction * 300
                        elif dist < grav_r * 0.5:
                            warning = "⚠ EXTREME GRAVITY ⚠"
                            
                elif isinstance(obj, Star):
                    danger_r = obj.scale_x * 2
                    if dist < danger_r * 0.5:
                        warning = "⚠ RADIATION WARNING ⚠"
                        
        self.warning.text = warning

effects = Effects()

# ══════════════════════════════════════════════════════════════════
# STAR CLASS (simplified)
# ══════════════════════════════════════════════════════════════════

class Star(Entity):
    def __init__(self, position, seed):
        self.seed = seed
        self.rng = random.Random(seed)
        
        props = get_star_type(self.rng)
        self.main_color = props['color']
        self.type_name = props['name']
        
        size = self.rng.uniform(*props['size'])
        
        super().__init__(
            model='sphere',
            color=self.main_color,
            position=position,
            scale=size,
            unlit=True
        )
        
        floating_origin.register(self)
        
        # Single glow layer only
        Entity(
            parent=self,
            model='sphere',
            scale=1.5,
            color=props['glow'].tint(-0.5),
            unlit=True
        )
        
        self.star_name = f"Star-{seed % 9999}"
        self.planets = []
        self.planets_loaded = False
        
    def load_planets(self):
        if self.planets_loaded:
            return
            
        num_planets = self.rng.randint(2, 6)  # Reduced max
        
        for i in range(num_planets):
            orbital_radius = self.scale_x * 3 + 200 + i * 150
            
            # Gas giant chance
            if i >= num_planets - 2 and self.rng.random() > 0.5:
                planet = GasGiant(self, orbital_radius, self.seed + i + 1000)
            else:
                planet = Planet(self, orbital_radius, self.seed + i + 1000)
                
            self.planets.append(planet)
            
        self.planets_loaded = True
        
    def unload_planets(self):
        for planet in self.planets:
            planet.cleanup()
        self.planets.clear()
        self.planets_loaded = False
        
    def cleanup(self):
        self.unload_planets()
        floating_origin.unregister(self)
        destroy(self)

# ══════════════════════════════════════════════════════════════════
# PLANET CLASS (simplified)
# ══════════════════════════════════════════════════════════════════

class Planet(Entity):
    def __init__(self, parent_star, orbital_radius, seed):
        self.parent_star = parent_star
        self.orbital_radius = orbital_radius
        self.rng = random.Random(seed)
        
        props = self.rng.choice(PLANET_TYPES)
        self.type_name = props['name']
        
        size = self.rng.uniform(*props['size'])
        
        self.orbital_speed = 0.3 / math.sqrt(orbital_radius / 100)
        self.orbital_angle = self.rng.uniform(0, math.pi * 2)
        self.rotation_speed = self.rng.uniform(20, 50)
        
        x = math.cos(self.orbital_angle) * orbital_radius
        z = math.sin(self.orbital_angle) * orbital_radius
        
        super().__init__(
            model='sphere',
            color=props['color'],
            position=parent_star.position + Vec3(x, 0, z),
            scale=size
        )
        
        floating_origin.register(self)
        
        # Single ring for some planets
        if size > 20 and self.rng.random() > 0.7:
            Entity(
                parent=self,
                model='circle',
                scale=2.2,
                rotation_x=80,
                color=color.white.tint(-0.5),
                double_sided=True
            )
            
    def update(self):
        if not self.parent_star:
            return
            
        dt = time.dt * effects.time_dilation
        self.orbital_angle += dt * self.orbital_speed
        
        x = math.cos(self.orbital_angle) * self.orbital_radius
        z = math.sin(self.orbital_angle) * self.orbital_radius
        
        self.position = self.parent_star.position + Vec3(x, 0, z)
        self.rotation_y += dt * self.rotation_speed
        
    def cleanup(self):
        floating_origin.unregister(self)
        destroy(self)

# ══════════════════════════════════════════════════════════════════
# GAS GIANT (simplified)
# ══════════════════════════════════════════════════════════════════

class GasGiant(Entity):
    def __init__(self, parent_star, orbital_radius, seed):
        self.parent_star = parent_star
        self.orbital_radius = orbital_radius
        self.rng = random.Random(seed)
        
        props = self.rng.choice(GAS_GIANTS)
        self.type_name = props['name']
        
        size = self.rng.uniform(*props['size'])
        
        self.orbital_speed = 0.2 / math.sqrt(orbital_radius / 100)
        self.orbital_angle = self.rng.uniform(0, math.pi * 2)
        self.rotation_speed = self.rng.uniform(30, 60)
        
        x = math.cos(self.orbital_angle) * orbital_radius
        z = math.sin(self.orbital_angle) * orbital_radius
        
        super().__init__(
            model='sphere',
            color=props['color'],
            position=parent_star.position + Vec3(x, 0, z),
            scale=size
        )
        
        floating_origin.register(self)
        
        # Rings (high chance)
        if self.rng.random() > 0.4:
            Entity(
                parent=self,
                model='circle',
                scale=2,
                rotation_x=75,
                color=color.orange.tint(-0.4),
                double_sided=True
            )
            
    def update(self):
        if not self.parent_star:
            return
            
        dt = time.dt * effects.time_dilation
        self.orbital_angle += dt * self.orbital_speed
        
        x = math.cos(self.orbital_angle) * self.orbital_radius
        z = math.sin(self.orbital_angle) * self.orbital_radius
        
        self.position = self.parent_star.position + Vec3(x, 0, z)
        self.rotation_y += dt * self.rotation_speed
        
    def cleanup(self):
        floating_origin.unregister(self)
        destroy(self)

# ══════════════════════════════════════════════════════════════════
# BLACK HOLE (simplified)
# ══════════════════════════════════════════════════════════════════

class BlackHole(Entity):
    def __init__(self, position, mass, seed):
        size = mass * 4
        
        super().__init__(
            model='sphere',
            color=color.black,
            position=position,
            scale=size
        )
        
        floating_origin.register(self)
        
        self.star_name = f"BlackHole-{seed % 999}"
        self.type_name = "Black Hole"
        self.planets_loaded = False
        self.planets = []
        
        # Simple glow
        Entity(
            parent=self,
            model='sphere',
            scale=1.3,
            color=color.violet.tint(-0.5),
            unlit=True
        )
        
        # Single accretion disk
        self.disk = Entity(
            parent=self,
            model='circle',
            scale=4,
            rotation_x=80,
            color=color.orange,
            double_sided=True,
            unlit=True
        )
        
    def update(self):
        self.disk.rotation_y += time.dt * 100
        
    def cleanup(self):
        floating_origin.unregister(self)
        destroy(self)
        
    def load_planets(self):
        pass
    def unload_planets(self):
        pass

# ══════════════════════════════════════════════════════════════════
# NEBULA (simplified)
# ══════════════════════════════════════════════════════════════════

class Nebula(Entity):
    COLORS = [color.pink, color.cyan, color.violet, color.orange, color.magenta]
    
    def __init__(self, position, size, seed):
        rng = random.Random(seed)
        nebula_color = rng.choice(self.COLORS)
        
        super().__init__(
            model='sphere',
            color=nebula_color.tint(-0.7),
            position=position,
            scale=size,
            unlit=True
        )
        
        floating_origin.register(self)
        self.star_name = f"Nebula-{seed % 999}"
        self.type_name = "Nebula"
        self.planets_loaded = False
        self.planets = []
        
        # Single inner layer
        Entity(
            parent=self,
            model='sphere',
            scale=0.7,
            color=nebula_color.tint(-0.5),
            unlit=True
        )
            
    def cleanup(self):
        floating_origin.unregister(self)
        destroy(self)
        
    def load_planets(self):
        pass
    def unload_planets(self):
        pass

# ══════════════════════════════════════════════════════════════════
# UNIVERSE CHUNKS
# ══════════════════════════════════════════════════════════════════

class UniverseChunk:
    def __init__(self, coords):
        self.coords = coords
        self.objects = []
        self.loaded = False
        
    def generate(self):
        if self.loaded:
            return
            
        cx, cy, cz = self.coords
        seed = get_seed(cx, cy, cz, "chunk")
        rng = random.Random(seed)
        
        # Generate stars
        num_stars = Config.STARS_PER_CHUNK
        
        for i in range(num_stars):
            star_seed = get_seed(cx, cy, cz, f"star{i}")
            star_rng = random.Random(star_seed)
            
            pos = Vec3(
                cx * Config.CHUNK_SIZE + star_rng.uniform(200, Config.CHUNK_SIZE - 200),
                cy * Config.CHUNK_SIZE + star_rng.uniform(-100, 100),
                cz * Config.CHUNK_SIZE + star_rng.uniform(200, Config.CHUNK_SIZE - 200)
            )
            
            star = Star(pos, star_seed)
            self.objects.append(star)
            
        # Rare nebula
        if rng.random() < 0.08:
            pos = Vec3(
                cx * Config.CHUNK_SIZE + rng.uniform(0, Config.CHUNK_SIZE),
                cy * Config.CHUNK_SIZE + rng.uniform(-200, 200),
                cz * Config.CHUNK_SIZE + rng.uniform(0, Config.CHUNK_SIZE)
            )
            nebula = Nebula(pos, rng.uniform(150, 350), seed + 10000)
            self.objects.append(nebula)
            
        # Rare black hole
        if rng.random() < 0.03:
            pos = Vec3(
                cx * Config.CHUNK_SIZE + rng.uniform(0, Config.CHUNK_SIZE),
                cy * Config.CHUNK_SIZE + rng.uniform(-50, 50),
                cz * Config.CHUNK_SIZE + rng.uniform(0, Config.CHUNK_SIZE)
            )
            bh = BlackHole(pos, rng.uniform(8, 20), seed + 20000)
            self.objects.append(bh)
            
        self.loaded = True
        
    def unload(self):
        for obj in self.objects:
            obj.cleanup()
        self.objects.clear()
        self.loaded = False


class Universe:
    def __init__(self):
        self.chunks = {}
        
    def get_chunk_coords(self, pos):
        return (
            int(math.floor(pos.x / Config.CHUNK_SIZE)),
            int(math.floor(pos.y / Config.CHUNK_SIZE)),
            int(math.floor(pos.z / Config.CHUNK_SIZE))
        )
        
    def update(self, player_pos):
        current = self.get_chunk_coords(player_pos)
        
        needed = set()
        for dx in range(-Config.RENDER_DISTANCE, Config.RENDER_DISTANCE + 1):
            for dz in range(-Config.RENDER_DISTANCE, Config.RENDER_DISTANCE + 1):
                if dx*dx + dz*dz <= Config.RENDER_DISTANCE**2:
                    needed.add((current[0]+dx, 0, current[2]+dz))
                        
        for coords in needed:
            if coords not in self.chunks:
                chunk = UniverseChunk(coords)
                chunk.generate()
                self.chunks[coords] = chunk
                
        to_remove = [c for c in self.chunks if c not in needed]
        for coords in to_remove:
            self.chunks[coords].unload()
            del self.chunks[coords]
            
        # Load/unload planets
        for chunk in self.chunks.values():
            for obj in chunk.objects:
                if isinstance(obj, Star):
                    dist = (obj.position - player_pos).length()
                    if dist < Config.PLANET_LOAD_DIST and not obj.planets_loaded:
                        obj.load_planets()
                    elif dist > Config.PLANET_UNLOAD_DIST and obj.planets_loaded:
                        obj.unload_planets()
                        
    def get_nearest(self, pos, max_dist=5000):
        nearest = None
        nearest_dist = max_dist
        
        for chunk in self.chunks.values():
            for obj in chunk.objects:
                d = (obj.position - pos).length()
                if d < nearest_dist:
                    nearest = obj
                    nearest_dist = d
                    
        return nearest, nearest_dist

universe = Universe()

# ══════════════════════════════════════════════════════════════════
# SPACESHIP
# ══════════════════════════════════════════════════════════════════

class Spaceship(Entity):
    def __init__(self):
        super().__init__()
        
        self.speed = 0
        self.mode = 0
        self.modes = ['Slow', 'Fast', 'Warp', 'Hyper']
        self.max_speeds = [30, 200, 2000, 15000]
        self.accels = [15, 100, 500, 3000]
        
        camera.parent = self
        camera.position = (0, 2, -8)
        camera.rotation_x = 5
        
        # Simple ship
        self.body = Entity(
            parent=self,
            model='cube',
            scale=(1, 0.3, 2),
            color=color.dark_gray,
            position=(0, 0, 3)
        )
        
        self.engine = Entity(
            parent=self.body,
            model='sphere',
            scale=0.3,
            color=color.cyan,
            position=(0, 0, -1.2),
            unlit=True
        )
        
        self.target = None
        self.autopilot = False
        
    def update(self):
        max_speed = self.max_speeds[self.mode]
        accel = self.accels[self.mode]
        dt = time.dt * effects.time_dilation
        
        # Mouse look
        if mouse.locked:
            self.rotation_y -= mouse.velocity[0] * 80
            self.rotation_x -= mouse.velocity[1] * 80
            self.rotation_x = clamp(self.rotation_x, -85, 85)
            
        # Roll
        if held_keys['q']:
            self.rotation_z += dt * 50
        if held_keys['e']:
            self.rotation_z -= dt * 50
            
        # Thrust
        if held_keys['w']:
            self.speed = min(self.speed + accel * dt, max_speed)
        elif held_keys['s']:
            self.speed = max(self.speed - accel * dt, -max_speed * 0.2)
        else:
            self.speed *= 0.99
            
        movement = self.forward * self.speed * dt
        
        # Strafe
        strafe = max(abs(self.speed) * 0.3, 15)
        if held_keys['a']:
            movement -= self.right * strafe * dt
        if held_keys['d']:
            movement += self.right * strafe * dt
        if held_keys['space']:
            movement += self.up * strafe * dt
        if held_keys['left shift']:
            movement -= self.up * strafe * dt
            
        # Gravity
        movement += effects.gravity_pull * dt
        
        self.position += movement
        
        # Autopilot
        if self.autopilot and self.target:
            direction = (self.target.position - self.position).normalized()
            target_y = math.degrees(math.atan2(direction.x, direction.z))
            target_x = math.degrees(math.asin(-direction.y))
            self.rotation_y = lerp(self.rotation_y, target_y, dt * 2)
            self.rotation_x = lerp(self.rotation_x, target_x, dt * 2)
            
        # Floating origin
        shift = floating_origin.update(self.position)
        if shift.length_squared() > 0:
            self.position -= shift
            background_stars.position -= shift
            
        # Engine color
        mode_colors = [color.cyan, color.lime, color.orange, color.magenta]
        self.engine.color = mode_colors[self.mode]
        self.engine.scale = 0.3 + (abs(self.speed) / max_speed) * 0.3
        
    def input(self, key):
        if key == '1':
            self.mode = 0
        elif key == '2':
            self.mode = 1
        elif key == '3':
            self.mode = 2
        elif key == '4':
            self.mode = 3
        elif key == 'i':
            self.mode = min(self.mode + 1, 3)
            self.speed = self.max_speeds[self.mode] * 0.7
        elif key == 'x':
            self.speed = 0
        elif key == 't':
            obj, dist = universe.get_nearest(self.position)
            self.target = obj
        elif key == 'y':
            self.autopilot = not self.autopilot
        elif key == 'r':
            self.position += self.forward * 200
            self.speed = 0

# ══════════════════════════════════════════════════════════════════
# HUD (simplified)
# ══════════════════════════════════════════════════════════════════

class HUD(Entity):
    def __init__(self, ship):
        super().__init__(parent=camera.ui)
        self.ship = ship
        
        self.info = Text(position=(-0.85, 0.45), scale=1.2, color=color.white)
        self.target_info = Text(position=(-0.85, 0.35), scale=1, color=color.yellow)
        self.near_info = Text(position=(-0.85, 0.25), scale=0.9, color=color.lime)
        
        Text(
            text='WASD:Fly | QE:Roll | 1-4:Speed | I:Boost | T:Target | Y:Auto | X:Stop | R:Jump',
            position=(0, -0.47),
            origin=(0, 0),
            scale=0.6,
            color=color.gray
        )
        
        # Crosshair
        Entity(parent=camera.ui, model='quad', scale=0.005, color=color.white)
        
    def update(self):
        speed = abs(self.ship.speed)
        mode = self.ship.modes[self.ship.mode]
        
        if speed > 1000:
            speed_str = f'{speed/1000:.1f}k'
        else:
            speed_str = f'{speed:.0f}'
            
        self.info.text = f'Speed: {speed_str} | Mode: {mode}'
        
        # Target
        if self.ship.target:
            dist = (self.ship.target.position - self.ship.position).length()
            auto = ' [AUTO]' if self.ship.autopilot else ''
            if dist > 1000:
                self.target_info.text = f'Target: {self.ship.target.star_name} | {dist/1000:.1f}k{auto}'
            else:
                self.target_info.text = f'Target: {self.ship.target.star_name} | {dist:.0f}{auto}'
        else:
            self.target_info.text = 'No target (T)'
            
        # Nearest
        obj, dist = universe.get_nearest(self.ship.position, 3000)
        if obj:
            if dist > 1000:
                self.near_info.text = f'Near: {obj.type_name} | {dist/1000:.1f}k'
            else:
                self.near_info.text = f'Near: {obj.type_name} | {dist:.0f}'
        else:
            self.near_info.text = ''

# ══════════════════════════════════════════════════════════════════
# WARP EFFECT (reduced particles)
# ══════════════════════════════════════════════════════════════════

class WarpEffect(Entity):
    def __init__(self, ship):
        super().__init__(parent=ship)
        self.ship = ship
        self.streaks = []
        
        for _ in range(30):  # Reduced from 100
            streak = Entity(
                parent=self,
                model='cube',
                scale=(0.01, 0.01, 0.5),
                position=(
                    random.uniform(-6, 6),
                    random.uniform(-6, 6),
                    random.uniform(10, 40)
                ),
                color=color.cyan,
                unlit=True,
                enabled=False
            )
            streak.start_z = streak.z
            self.streaks.append(streak)
            
    def update(self):
        mode = self.ship.mode
        speed = abs(self.ship.speed)
        max_speed = self.ship.max_speeds[mode]
        ratio = speed / max_speed if max_speed > 0 else 0
        
        active = mode >= 2 and ratio > 0.2
        
        for streak in self.streaks:
            streak.enabled = active
            if active:
                streak.color = color.magenta if mode == 3 else color.cyan
                streak.z -= time.dt * 80 * ratio
                streak.scale_z = 0.5 + ratio * 8
                
                if streak.z < -5:
                    streak.z = streak.start_z
                    streak.x = random.uniform(-6, 6)
                    streak.y = random.uniform(-6, 6)

# ══════════════════════════════════════════════════════════════════
# START
# ══════════════════════════════════════════════════════════════════

ship = Spaceship()
ship.position = Vec3(2000, 0, 2000)

hud = HUD(ship)
warp = WarpEffect(ship)

mouse.locked = True
window.fps_counter.enabled = True
window.fullscreen = True

Text(
    text='UNIVERSE SIMULATOR',
    position=(0, 0.47),
    origin=(0, 0),
    scale=1.2,
    color=color.white
)

def update():
    universe.update(ship.position)
    effects.update(ship.position)

def input(key):
    if key == 'escape':
        mouse.locked = not mouse.locked

print("\n" + "="*50)
print("         UNIVERSE SIMULATOR (OPTIMIZED)")
print("="*50)
print("  WASD       - Fly")
print("  Q/E        - Roll")
print("  Space/Shift- Up/Down")
print("  1-4        - Speed modes")
print("  I          - INSTANT BOOST")
print("  T          - Target nearest")
print("  Y          - Autopilot")
print("  X          - Stop")
print("  R          - Emergency jump")
print("="*50)
print("  Stars appear as specks - press I to boost!")
print("="*50 + "\n")

app.run()