from astra_world.rendering import WorldRenderer
from astra_world.world_builder import build_world
from astra_world.world_spec import WorldSpec
from PIL import Image
from io import BytesIO


def test_actual_mujoco_frame_is_jpeg():
    world = build_world(
        WorldSpec(
            name="Render test",
            entities=[
                dict(
                    id="red", asset_id="small_box", color="red", position=[0.4, 0, 0.02]
                )
            ],
        )
    )
    renderer = WorldRenderer(width=480, height=320)
    try:
        frame = renderer.render(world)
        im = Image.open(BytesIO(frame))
        assert im.size == (480, 320)
        assert im.getextrema()[0][1] > im.getextrema()[0][0]
    finally:
        renderer.close()
