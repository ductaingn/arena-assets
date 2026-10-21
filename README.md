# arena-assets

Build, query, and manage 3D model databases for [Arena](https://github.com/Arena-Rosnav).

`arena-assets` converts datasets of assets (objects, materials, humans, and sounds) into a searchable database: models are converted with Blender, optionally baked with Isaac Sim, and indexed in [ChromaDB](https://www.trychroma.com/) so they can be queried by natural-language description.

## Installation

```sh
pip install ./src
```

Note: if you want to install `arena_assets[build]`, you must use Python 3.11.* (bpy dependency).

The repository is also an `ament_cmake` ROS package; building it in a colcon workspace installs the CLI into a dedicated venv.

## Usage

### Fetch prebuilt assets

```sh
# list available assets
arena-assets net default list

# download assets into a local directory
arena-assets net default fetch -o ./models Common/Material/ABS_Hard_Leather

# only download specific model formats, skip annotations
arena-assets net default fetch -o ./models --format usdz --no-annotation
```

`default` resolves to the public GCS bucket; any other bucket name can be passed instead.

### Build a database

```sh
arena-assets db ./models build -i ./dataset
```

Build options are passed with `-o key=value`:

| option | asset types | effect |
| --- | --- | --- |
| `formats` | object | comma-separated model formats to export (`*` for all) |
| `previews` | object | render preview images |
| `procthor` | object, material | export a ProcTHOR-compatible `asset-database.json` |
| `bake-mdl` | object | bake embedded MDL materials with Isaac Sim |
| `recompute-affordances` | object | recompute the z range of every affordance slot, hand-set ones included |

Each type is discovered under its own directory of the dataset (`Object`, `Material`, `Human`, `Sound`).

`--overwrite` controls re-building: `skip` (default) keeps existing entries, `overwrite` rebuilds them, `annotations` only refreshes annotations.

### Object lights

An object's `annotation.yaml` may carry `lights:`, one entry per light it emits: `name` (default `light`), `fixture` (`bulb`, `spot`, `downlight`, `panel` or `tube`), `pose` in the frame of the built model as `[[x, y, z], [w, x, y, z]]` or `[[x, y, z], [roll, pitch, yaw]]` (the light emits along the local -z axis of that pose, so `[1, 0, 0, 0]` shines straight down), `lumens`, `cct_K` (default 6500) and, for a spot, `cone_deg`. The build fills it from the lights of the source model (point lights become bulbs, spots keep their cone, area lights become panels or downlights, suns are skipped, 683 lm per Blender watt, color temperature from the light color) unless the source annotation already lists lights, and strips the lights from every exported model, so the annotation is the only place an object's light lives. glTF sources carry point and spot lights, USD sources also carry area lights.

A light may also carry `glow`, the name of the Blender material whose surfaces glow while the light is on. The build gives each extracted light the nearest emissive material: one whose Principled BSDF has a constant, non-black `Emission Color` and an `Emission Strength` above 0 (a textured emission never counts), measured from the light to the center of the bounding box of the faces using the material. Several lights may share one material, and an emissive material nearest to no light is left as it is. A hand-annotated `glow` must name a material used by the model, otherwise the build of that object fails with the list of its materials. Glow materials ship dark: their emission is cleared in every exported format, and the simulator lights them with the lamp. In the SDF export each glow material is a visual of its own, `glow_<material>` with the mesh `<name>.glow_<i>.dae` (`i` counts the glow materials in sorted name order), next to `visual` with the remaining faces in `<name>.dae`. A glow visual carries a `<material>` with the base color as `ambient` and `diffuse` and a black `specular` and `emissive`, unless the base color is textured.

### Object affordances

An object's `annotation.yaml` may carry `affordances:`, a map from an affordance kind to its slots. The only kind is `sit`, and an unknown kind fails the build of that object. A slot is a `bbox` over the seat surface in the frame of the built model, `[[x_min, x_max], [y_min, y_max]]`, and a `yaw`: the directions a person sitting there may face, an angle in radians, an axis (`+x`, `+y`, `-x`, `-y` or `xy`, as for `face`) or a list of them. A body rests anywhere inside the bbox and the legs hang over its edge, so each yaw points at an open edge of the seat. Slots of one object may overlap. Slots parse with `arena_simulation_setup.utils.affordance`, which the Arena workspace provides, so a standalone install reads and builds objects with affordances only next to it and lists such objects without their affordances.

```yaml
affordances:
  sit:
    - {bbox: [[-0.46, 0.48], [0.03, 0.72]], yaw: -y}
    - {bbox: [[-1.30, -0.56], [-1.13, -0.26]], yaw: [+x, -y]}
    - {}
```

The build adds a z range to each slot, from the floor of the object's bounding box up to the height of the seat surface seen from above inside the bbox: on a 2 cm grid of the model's top view, the 2 cm height bin between 0.2 and 0.9 m that covers the most cells, as the median of the cells within 5 cm of it. Without such a surface it takes `min(0.45, object height)` and warns. A slot whose source bbox already has a z range keeps it unless the build runs with `-o recompute-affordances`. A slot without a bbox, `{}` or `{yaw: ...}`, gets a square of up to 0.45 m clipped to the object, centered on the seat surface covering the most of the top view (the bbox center when there is none), facing its `yaw` or else `face`. At most one slot per kind may leave its bbox to the build. `--overwrite annotations` copies the z ranges of the last build, matching slots by their x and y ranges and the slot without a bbox to the one built slot no source bbox matches, when it faces the same way, and places the remaining slots (all of them with `-o recompute-affordances`) on the model the last build left in the database, so no model is exported again. Without that model, or when placing fails, the slots stay unplaced and the build warns.

`ObjectAnnotation.query_affordance(kind, slot)` returns a placed slot, one with a bbox and a yaw, as an `AffordanceSlot` and `affordance_slots()` returns every placed one. The slot schema (`Slot`, `structure_affordances`), the placed `AffordanceSlot` and the claims on them all live in `arena_simulation_setup.utils.affordance`, so the build and the runtime read slots the same way: an `AffordanceClaims`, one per placed object, keeps no two overlapping affordance boxes occupied at the same time, across all kinds. `available(kind)` lists the slots that overlap no occupied one, `closest(xy, kind)` returns the available slot nearest to an object-frame point (by distance to its box, 0 inside), and `claim` refuses a slot whose box overlaps an occupied one. Build it as `AffordanceClaims(annotation.affordance_slots())`, or as `AffordanceClaims.parse(affordances)` from the `affordances` block of a built `annotation.yaml`. A slot without a z range overlaps through all heights.

`arena-assets semantic seats` proposes sit slots with Gemini (install `arena_assets[vlm]` and set `GEMINI_API_KEY`):

```sh
arena-assets semantic seats propose ./dataset --renders ./models --out seats.yaml [--only Residential/Object/Sofa_B]...
arena-assets semantic seats apply seats.yaml ./dataset [--overwrite]
```

`propose` first asks which objects are for sitting, by asset path and description in batches of 50, unless `--only` names the assets. For each of them it sends the top-down and perspective previews of a build with `-o previews` and the worked example [seats_example.yaml](src/arena_assets/semantic/seats_example.yaml), asks `--samples` times (default 3) and keeps the slot count most answers agree on, the median box of matched slots and the facings most answers name. The example object is asked without its own example. It prints the tokens used and merges each object's proposal into `--out` as soon as it has it, keeping the objects already there. It stops at the first object whose samples all fail, the proposals so far kept. `apply` writes reviewed proposals into the source annotations, skipping objects that already have `sit` slots unless `--overwrite`, with which an empty list removes them. It checks every proposal before writing any file.

### Sound assets

A sound asset is a directory `Sound/<Name>/` holding its wav files and a `<Name>.yaml` manifest (version 2: `kind`, `level_db`, `loop`, `variants`, optional `desc` and `tags`). The manifest is the only annotation source: the build derives `annotation.yaml` from it (tags are the manifest tags, every variant's tags, and the kind) and fails on a manifest with the wrong version or without `kind`, `level_db` or `variants`.

```sh
arena-assets db ./sounds build -t sound -i ./dataset
arena-assets db ./sounds query sound "robot motor hum" --filter "level_db>50"
arena-assets db ./sounds query sound "steps on wood" --filter "kind=footstep" -n 3
```

`kind`, `loop`, `level_db`, and the comma-joined variant ids are stored as metadata.

### Upload a built database

```sh
arena-assets net my-bucket author ./models -d Common
```

Files already present in the bucket with the same size are skipped. Uploading requires write access to the bucket: a token is taken from `--token`, the `GCS_ACCESS_TOKEN` environment variable, or `gcloud auth print-access-token`, in that order.

### Query

```sh
# best match (prints the asset path)
arena-assets db ./models query material "light brown wood"

# top 5 matches with distance scores
arena-assets db ./models query object "office chair" -n 5 --scores

# constrain by size (width/depth/height/volume in meters, repeatable)
arena-assets db ./models query object "office chair" --filter "height<1.0" --filter "volume>0.05"

# list everything of a type
arena-assets db ./models list material
```

All logs and progress bars go to stderr; stdout only carries the results, so output can be piped.

### Splat walls and objects

`arena-assets splat` turns a trained 3D Gaussian splat of a real wall into an object bundle that a wall kind places with its `model:` directive. With `--object` it frames a free-standing object instead, see [Splat objects](#splat-objects). The recipe for a wall, from capture to a pushed bundle:

1. Capture a phone video or photos of the wall area and run COLMAP on them (3DGRUT reads COLMAP layouts).
2. Train with 3DGRUT and export OpenUSD:

   ```sh
   python train.py --config-name apps/colmap_3dgut_mcmc.yaml path=<scene> out_dir=<out> experiment_name=<name> \
       dataset.downsample_factor=2 strategy.add.max_n_gaussians=4000000 \
       export_usd.enabled=true export_usd.export_cameras=false export_usd.export_background=false
   ```

   4M Gaussians at downsample factor 2 gave the best look, about 20 min on an RTX PRO 6000. Export without cameras and background, because Isaac's SpawnPrims turns the default light off for assets that carry lights.
3. Register and crop the export into a bundle directory that the object resolver finds, for example a world's `assets/Common/Object/<name>`:

   ```sh
   arena-assets splat <out>/<name>/<run>/export_last_lightfield.usdz ./assets/Common/Object/<name> --points <name>.txt --width 3.2 --height 2.5
   ```

   Registration, crop and `--drop` boxes are given in the wall frame: meters, origin at the base midpoint of the captured wall, X along the wall, Y facing into the room and Z up, with Y = 0 on the captured wall plane. `--points FILE` fits the registration (uniform scale, rotation, translation) from at least 3 non-collinear correspondences, one `sx sy sz tx ty tz` per line (a point in the source stage and the same point in the wall frame), and prints the fitted scale and the RMS residual in meters. `--matrix` takes a known transform instead: 16 numbers in one quoted argument, row-major, `p_target = M @ p_stage`.

   The crop keeps X within `--width`, Z from `--floor` (default 0.03) to `--height`, and Y from `-behind` (default 0) to `--depth` (default 0.05). Captured geometry is seen by Isaac lidar and depth cameras but has no collision, so the default keeps only 5 cm in front of the wall plane and deeper crops are opt-in. `--drop 'XMIN YMIN ZMIN XMAX YMAX ZMAX'` (repeatable) removes the Gaussians inside a wall-frame box. `--max-scale` (default 0.2 m) removes oversized Gaussians, which would blow up the extent.

   The bundle origin is the wall base midpoint at the back of the crop: the bundle is the wall frame shifted by `+behind` along Y, and the physical wall of the wall kind sits at its Y = 0, so the whole capture lies in front of that wall. With the default `--behind 0` nothing behind the captured wall plane is kept. With `--behind B` the captured wall surface stands B meters in front of the physical wall, seen by lidar and depth cameras but not solid.

   The bundle holds `<name>.gaussians.usdc` (the kept Gaussians), `<name>.usda` (a Z-up wrapper in meters that carries the registration), `annotation.yaml` and, with `--collider-mesh`, the collider (see [Splat colliders](#splat-colliders)).
4. The rule: anything standing out from the wall must be wholly inside or wholly outside the crop, otherwise its cut parts float. A floor plant whose leaves reached into the crop had to be taken out with `--drop`.
5. Publish with `arena asset push object <name>`.
6. Place it in a wall kind:

   ```yaml
   main:
     - material: Plaster_Wall      # the wall: collision, lidar, map, and the backdrop behind the capture
       height: 2.6
       shadows: false
     - model: <name>
       at: 50%
   ```

   - The segment is an ordinary drawn wall on the wall line.
   - It shows through holes in the capture.
   - `shadows: false` keeps simulator shadows off the capture, whose lighting is baked in.

   Isaac only: Gazebo and the other simulators skip the USD-only object and draw the wall.

#### Splat objects

`--object` frames a free-standing object, captured from all sides, as a bundle that worlds and scenarios place like any other static object. Capture and training follow steps 1 and 2 above, then:

```sh
arena-assets splat <export>.usdz ./assets/Common/Object/<name> --object --matrix '<16 numbers>' --width 1.2 --depth 1.2 --height 1.1
```

- Registration, crop and `--drop` boxes are given in the object frame: meters, origin on the floor under the object center, X and Y level with Y toward the object front, Z up. `--points` and `--matrix` work as for walls.
- The crop keeps X within `--width`, Y within `--depth` (required), both centered on the origin, and Z from `--floor` (default 0.03, which cuts the captured floor away) to `--height`. `--behind` applies to walls only.
- The bundle origin is the object frame origin, so a pose at floor height stands the object on the floor, and `annotation.yaml` records the crop as the bounding box.

Place it like any other object, in a world level's or a scenario's `objects:` list:

```yaml
objects:
  - {name: garden_table, model: <name>, pose: [5.0, 4.0, 0.0]}
```

Without a collider the object is visual only. Isaac cameras, lidar and depth cameras see it, physics does not: robots drive through it. The task generator keeps spawns and goals off its bounding box, and humansim receives the box as an obstacle. The capture's lighting is baked in. Isaac only: Gazebo and the other simulators skip it.

#### Splat colliders

`--collider-mesh FILE` gives a splat bundle a collider at build time. FILE is a triangle mesh of the same capture, `.obj` or `.ply`, in the source stage frame of the export, for example a surface reconstructed from the training views. Building that mesh is outside this tool. It is registered with the same `--matrix` or `--points` as the Gaussians and cropped by the same box: a triangle stays only with all its corners inside the crop and none inside a `--drop` box, and the verb prints how many it keeps. The kept triangles are baked into `<name>.usda` as described under [Colliders](#colliders). The bundle has no SDF of its own, so the verb also writes `<name>.sdf/<name>.sdf` holding only the collision: Gazebo then has the obstacle without the capture.

### Colliders

An object's `annotation.yaml` may name a collider mesh under `collider:`. It replaces the object's collision in every model the bundle holds, while cameras, lidar and depth cameras keep seeing the visual model:

- USD (`<name>.usdz`, `.usd`, `.usda` or `.usdc`): an invisible static triangle-mesh collider `collider/mesh` under the default prim, with the collision of the visual meshes switched off (`physics:collisionEnabled = false`). Without one, the build gives every visual mesh a static triangle-mesh collision, as the SDF has.
- SDF (`<name>.sdf/<name>.sdf`): the only `<collision>`, named `collider`, with the mesh `<name>.collider.obj` next to the SDF. An SDF without one collides with its visual meshes.
- The mesh itself stays in the bundle as `<name>.collider.obj`, which the built annotation names.

In a build dataset, `collider:` names an `.obj` or `.ply` triangle mesh in the object's source directory, in the frame of the source model in meters, Z up. The build moves the mesh with the model when it stands the model upright on the floor, then bakes it into every exported format.

`arena-assets collider BUNDLE MESH` does the same for a built bundle directory, for example a pulled object or a world-local copy:

```sh
arena-assets collider ./assets/Common/Object/<name> <hull>.obj
arena-assets collider ./assets/Common/Object/<name> --remove
```

- MESH is an `.obj` or `.ply` triangle mesh (ASCII or binary, polygons are split into triangles) in the bundle frame, the frame of the bundle's models in meters, Z up. `--matrix` maps a mesh from another frame into it.
- It bakes into the models present in the directory and prints them. A bundle pulled with only some formats gets the collider only in those.
- Running it again replaces the collider. `--remove` takes it out of the models, which then collide as they were built, and deletes `<name>.collider.obj` and the annotation entry.
- `arena asset push object <name>` publishes the changed models with the bundle.

## If your materials are missing in the output model

USD files can contain embedded MDL materials, which need to be baked before Blender can bake them correctly.
You can specify `-o bake-mdl=/path/to/isaacsim/python.sh` to enable automatic baking of MDL materials during database build.
Specifying `-o bake-mdl` without a path uses a docker image from the [Nvidia nvcr registry](https://nvcr.io/nvidia/isaac-sim). (May not work consistently.)

## Development

```sh
pip install "./src[test]"
pre-commit install

pytest tests/
ruff check . && ruff format src
```
